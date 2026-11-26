# the three-stage model of the A549 experiments, rebuilt for NCM460 and for loading A549 checkpoints; the model classes come from pdgraph
import torch

import config as cfg
from pdgraph.appearance import AppearanceExpert
from pdgraph.gate import assemble_gate_model
from pdgraph.graph_expert import GraphClassifier
from pdgraph.utils import get_device


def device():
    return get_device(cfg.DEVICE_INDEX)


def build_stage1():
    return AppearanceExpert(num_classes=cfg.NUM_CLASSES, embed_dim=cfg.STAGE1["embed_dim"])


def build_stage2():
    s = cfg.STAGE2
    return GraphClassifier(num_classes=cfg.NUM_CLASSES, in_channels=len(cfg.CHANNELS), embed_dim=s["embed_dim"], patch_stride=s["patch_stride"],
                           depth=s["depth"], num_heads=s["num_heads"], dropout=s["dropout"])


def build_stage3(app_ckpt=None, graph_ckpt=None):
    # gate model with frozen experts; without checkpoints the experts are randomly initialised
    return assemble_gate_model(build_stage1(), build_stage2(), cfg.NUM_CLASSES, cfg.STAGE2["embed_dim"], cfg.STAGE3, app_ckpt, graph_ckpt)


def experts_equal(model_a, model_b):
    # True when two Stage 3 models have identical expert weights (only their gates may differ), so one pass over the data serves both
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    return all(torch.equal(state_a[k], state_b[k]) for k in state_a if not (k.startswith("gate_head.") or k == "prior_beta_raw"))


def load_stage3_checkpoint(path):
    # complete Stage 3 model (both experts and the gate) from a best_model.pth of a Stage 3 run, in eval mode
    model = build_stage3()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device()).eval()
