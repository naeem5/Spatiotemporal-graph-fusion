import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pdgraph.appearance import AppearanceExpert
from pdgraph.graph_expert import GRAPH_STAT_KEYS, stats_to_vector
from pdgraph.temporal_experts import graph_classifier_class


def top1_margin_from_logits(logits):
    prob = torch.softmax(logits, dim=1)
    top2 = torch.topk(prob, k=2, dim=1).values
    return top2[:, 0:1], top2[:, 0:1] - top2[:, 1:2]


def confidence_prior_logit(pmax, tau=0.75, alpha=10.0):
    # positive when the appearance expert is less confident than tau, negative when it is more confident
    return alpha * (tau - pmax)


def normalized_entropy(prob, eps=1e-8):
    return -(prob * prob.clamp_min(eps).log()).sum(dim=1, keepdim=True) / math.log(prob.size(1))


class GateHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        # the last layer starts at zero, so training begins from the confidence prior alone
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_gate):
        return self.net(z_gate)


class AppearanceGraphGateModel(nn.Module):
    # final logits = appearance logits + delta_scale * gate * (centred graph logits); only the gate is trained
    def __init__(self, appearance_model, graph_model, num_classes=10, graph_embed_dim=128, delta_scale=0.3, gate_tau=0.75,
                 gate_alpha=10.0, gate_mode="classwise", gate_hidden=64, gate_dropout=0.1, prior_beta_init=0.5):
        super().__init__()
        self.appearance_model = appearance_model
        self.graph_model = graph_model

        if gate_mode not in {"scalar", "classwise"}:
            raise ValueError("gate_mode must be 'scalar' or 'classwise'")
        self.gate_mode = gate_mode
        graph_in_dim = graph_embed_dim + len(GRAPH_STAT_KEYS)
        # gate input: graph features, both probability vectors, their absolute difference, and confidence, margin and
        # entropy of both experts
        gate_in_dim = graph_in_dim + 3 * num_classes + 6
        gate_out_dim = num_classes if gate_mode == "classwise" else 1
        self.gate_head = GateHead(in_dim=gate_in_dim, out_dim=gate_out_dim, hidden=gate_hidden, dropout=gate_dropout)

        if prior_beta_init <= 0:
            raise ValueError("prior_beta_init must be positive")
        # strength of the confidence prior, kept positive through a softplus
        self.prior_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(prior_beta_init)), dtype=torch.float32))

        self.delta_scale = delta_scale
        self.gate_tau = gate_tau
        self.gate_alpha = gate_alpha

    def train(self, mode=True):
        # both experts stay in eval mode, only the gate follows the mode
        super().train(mode)
        self.appearance_model.eval()
        self.graph_model.eval()
        self.gate_head.train(mode)
        return self

    def expert_features(self, x_app, x_graph, mask_graph):
        # outputs of the two frozen experts (no gradients); they never change during Stage 3, so they can be computed once and stored
        with torch.no_grad():
            app_pooled, logits_app = self.appearance_model.encode(x_app)
            graph_logits, graph_embed, stats, dev_stats = self.graph_model(x_graph, mask_graph)
            dev_vec = stats_to_vector(dev_stats)
            z_graph = self.graph_model.prepare_features(graph_embed, dev_stats)
        return dict(logits_app=logits_app, app_pooled=app_pooled, graph_logits=graph_logits, graph_embed=graph_embed, dev_vec=dev_vec, z_graph=z_graph)

    def forward(self, x_app, x_graph, mask_graph, return_feat=False):
        return self.fuse(self.expert_features(x_app, x_graph, mask_graph), return_feat)

    def fuse(self, features, return_feat=False):
        # gate and fusion from the outputs of expert_features; same return values as forward()
        logits_app, graph_logits = features["logits_app"], features["graph_logits"]
        app_pooled, graph_embed, dev_vec, z_graph = features["app_pooled"], features["graph_embed"], features["dev_vec"], features["z_graph"]
        with torch.no_grad():
            pmax, margin = top1_margin_from_logits(logits_app)
            app_prob = torch.softmax(logits_app, dim=1)
            app_entropy = normalized_entropy(app_prob)
            graph_prob = torch.softmax(graph_logits, dim=1)
            graph_pmax, graph_margin = top1_margin_from_logits(graph_logits)
            graph_entropy = normalized_entropy(graph_prob)
            # centre the graph logits: softmax ignores a shared offset, an additive correction must not carry it
            delta_logits = graph_logits - graph_logits.mean(dim=1, keepdim=True)

        z_gate = torch.cat([z_graph, app_prob, graph_prob, (app_prob - graph_prob).abs(), pmax, margin, app_entropy,
                            graph_pmax, graph_margin, graph_entropy], dim=1)
        learned_logit = self.gate_head(z_gate)
        prior_logit = confidence_prior_logit(pmax, tau=self.gate_tau, alpha=self.gate_alpha)
        prior_beta = F.softplus(self.prior_beta_raw)
        g_learned = torch.sigmoid(learned_logit)
        g_prior = torch.sigmoid(prior_beta * prior_logit)
        # the prior is added in logit space, so graph evidence can override a confident but wrong appearance prediction
        g = torch.sigmoid(learned_logit + prior_beta * prior_logit)

        logits_final = logits_app + self.delta_scale * g * delta_logits

        if return_feat:
            return (logits_final, logits_app, delta_logits, g, graph_embed, dev_vec, app_pooled, pmax, margin, g_learned,
                    g_prior, graph_prob, graph_pmax, graph_margin)
        return logits_final, logits_app, delta_logits, g, g_learned, g_prior


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def assemble_gate_model(appearance_model, graph_model, num_classes, graph_embed_dim, gate_settings, app_ckpt=None, graph_ckpt=None, delta_scale=None):
    # gate model on two frozen experts, optionally loaded from their Stage 1 and Stage 2 checkpoints;
    # gate_settings: delta_scale, gate_tau, gate_alpha, gate_hidden, gate_dropout, prior_beta_init
    for model, ckpt in ((appearance_model, app_ckpt), (graph_model, graph_ckpt)):
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
        freeze(model)
    s = gate_settings
    return AppearanceGraphGateModel(appearance_model, graph_model, num_classes=num_classes, graph_embed_dim=graph_embed_dim,
                                    delta_scale=s["delta_scale"] if delta_scale is None else delta_scale, gate_tau=s["gate_tau"],
                                    gate_alpha=s["gate_alpha"], gate_mode="classwise", gate_hidden=s["gate_hidden"],
                                    gate_dropout=s["gate_dropout"], prior_beta_init=s["prior_beta_init"])


def build_appearance_model(ckpt_path, num_classes, embed_dim=256):
    model = AppearanceExpert(num_classes=num_classes, embed_dim=embed_dim)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True), strict=True)
    return freeze(model)


def build_graph_model(ckpt_path, num_classes, in_channels=3, embed_dim=128, patch_stride=8, depth=2, num_heads=4, dropout=0.15, graph_arch="graph"):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt_channels = int(state["encoder.backbone.net.0.weight"].shape[1])
    if ckpt_channels != in_channels:
        raise ValueError(f"the Stage 2 checkpoint expects {ckpt_channels} channel(s) but {in_channels} were given; "
                         "use the same channels in the same order as in Stage 2")
    model = graph_classifier_class(graph_arch)(num_classes=num_classes, in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride,
                                               depth=depth, num_heads=num_heads, dropout=dropout)
    model.load_state_dict(state, strict=True)
    return freeze(model)
