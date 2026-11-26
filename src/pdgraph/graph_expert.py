# Stage 2: graph expert. Image patches of all frames are nodes; spatial edges join the 8 neighbours inside a frame,
# temporal edges join the same location across frames. Outputs a recording embedding and four descriptors.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatureMap(nn.Module):
    # small conv backbone: one frame -> grid of node vectors, input (N, C, H, W), output (N, D, H', W')
    def __init__(self, in_channels: int = 1, embed_dim: int = 128, patch_stride: int = 8):
        super().__init__()
        assert patch_stride in (4, 8, 16), "patch_stride must be 4, 8 or 16"

        # BatchNorm is kept because a GroupNorm variant performed worse in early tests. Padded frames never reach it:
        # the encoder removes them first, so they cannot disturb the running statistics.
        layers = [
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
        ]
        chan = 64
        if patch_stride >= 8:
            layers += [
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            ]
            chan = 128
        if patch_stride >= 16:
            layers += [
                nn.Conv2d(chan, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.SiLU(inplace=True),
            ]
            chan = 128

        layers += [nn.Conv2d(chan, embed_dim, 1, bias=False), nn.BatchNorm2d(embed_dim), nn.SiLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (N, D, H', W')


class LocalSpatialAttention(nn.Module):
    # attention over the k x k neighbourhood of every node (k = 3: the 8 neighbours), implemented with F.unfold
    def __init__(self, dim: int, k: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        assert k % 2 == 1, "k must be odd so every node has a well-defined center"
        self.k = k
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.kv_proj = nn.Conv2d(dim, dim * 2, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(1, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W = x.shape
        residual = x
        xn = self.norm(x)

        q = self.q_proj(xn)                     # (B,D,H,W)
        kv = self.kv_proj(xn)                    # (B,2D,H,W)
        k_, v_ = kv.chunk(2, dim=1)               # each (B,D,H,W)

        pad = self.k // 2
        KK = self.k * self.k
        N = H * W

        k_unf = F.unfold(k_, kernel_size=self.k, padding=pad)  # (B, D*KK, N)
        v_unf = F.unfold(v_, kernel_size=self.k, padding=pad)  # (B, D*KK, N)

        q = q.flatten(2).permute(0, 2, 1)                       # (B,N,D)
        k_unf = k_unf.view(B, D, KK, N).permute(0, 3, 2, 1)      # (B,N,KK,D)
        v_unf = v_unf.view(B, D, KK, N).permute(0, 3, 2, 1)      # (B,N,KK,D)

        q = q.view(B, N, self.num_heads, self.head_dim)                   # (B,N,h,hd)
        k_unf = k_unf.view(B, N, KK, self.num_heads, self.head_dim)       # (B,N,KK,h,hd)
        v_unf = v_unf.view(B, N, KK, self.num_heads, self.head_dim)       # (B,N,KK,h,hd)

        attn = torch.einsum("bnhd,bnkhd->bnhk", q, k_unf) * self.scale     # (B,N,h,KK)
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.einsum("bnhk,bnkhd->bnhd", attn, v_unf)                # (B,N,h,hd)
        out = out.reshape(B, N, D).permute(0, 2, 1).reshape(B, D, H, W)
        out = self.out_proj(out)

        return residual + self.drop(out)


class TemporalNodeAttention(nn.Module):
    # attention across time at every grid location (temporal edges); padded frames are never attended to
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False):
        B, T, H, W, D = x.shape
        residual = x
        xn = self.norm(x)

        # Fold the spatial grid into the batch dimension: every (b, h, w) location gets its own attention over time.
        x_flat = xn.permute(0, 2, 3, 1, 4).reshape(B * H * W, T, D)

        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B * H * W, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*H*W,h,T,T)

        mask_k = mask[:, None, None, :].expand(B, H * W, 1, T).reshape(B * H * W, 1, 1, T)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask_k, neg_inf)

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v)                                  # (B*H*W,h,T,hd)
        out = out.transpose(1, 2).reshape(B * H * W, T, D)
        out = self.proj(out)
        out = self.drop(out)

        out = out.view(B, H, W, T, D).permute(0, 3, 1, 2, 4)         # back to (B,T,H,W,D)
        result = residual + out
        # A padded query could still attend to valid keys and end up non-zero. Set padded frames back to exactly zero so
        # that nothing leaks from them into the next block.
        result = result * mask[:, :, None, None, None].to(result.dtype)

        if return_attn:
            # (B*H*W, h, T, T) -> (B, H, W, h, T, T): a soft T x T "adjacency matrix" per grid location and head.
            # Entry [query, key] is how much time point `query` attends to time point `key`.
            attn_out = attn.view(B, H, W, self.num_heads, T, T)
            return result, attn_out
        return result


GRAPH_STAT_KEYS = ("disruption", "order", "persistence", "recovery")


class GraphTransitionEncoder(nn.Module):
    # input x (B, C, T, H, W) and mask (B, T) (False = padded frame); output graph embedding (B, D) and the 4 descriptors
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        patch_stride: int = 8,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        spatial_k: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = PatchFeatureMap(in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride)

        self.spatial_layers = nn.ModuleList(
            [LocalSpatialAttention(embed_dim, k=spatial_k, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.temporal_layers = nn.ModuleList(
            [TemporalNodeAttention(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )

        self.node_norm = nn.LayerNorm(embed_dim)
        self.readout_pool = nn.Linear(embed_dim, 1)
        # The readout sees the mean, first, last and last-minus-first frame vectors, so the embedding depends on the order of
        # the frames (a plain mean over time would not).
        self.temporal_readout = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU(),
        )

    @staticmethod
    def _temporal_position_encoding(T: int, D: int, device, dtype):
        # Parameter-free sinusoidal encoding, valid for any sequence length.
        position = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(D, 1))
        )
        pe = torch.zeros(T, D, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        if D > 1:
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        return pe.to(dtype=dtype)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attn: bool = False, return_node_feat: bool = False):
        B, C, T, H, W = x.shape

        if mask.shape != (B, T):
            raise ValueError(f"mask must have shape {(B, T)}, got {tuple(mask.shape)}")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid frame")

        # Run the convolutional backbone on real frames only; zero-padded frames would distort the BatchNorm statistics.
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        valid_flat = mask.reshape(B * T)
        valid_feat = self.backbone(frames[valid_flat])
        _, D, Hp, Wp = valid_feat.shape
        feat = valid_feat.new_zeros(B * T, D, Hp, Wp)
        feat[valid_flat] = valid_feat
        node_feat = feat.view(B, T, D, Hp, Wp).permute(0, 1, 3, 4, 2)

        pos = self._temporal_position_encoding(T, D, x.device, node_feat.dtype)
        node_feat = node_feat + pos[None, :, None, None, :] * mask[:, :, None, None, None].to(node_feat.dtype)

        # Divided space-time blocks: at every depth, local spatial attention is followed by temporal attention.
        temporal_attns = [] if return_attn else None
        for spatial_layer, temporal_layer in zip(self.spatial_layers, self.temporal_layers):
            spatial_in = node_feat.permute(0, 1, 4, 2, 3).reshape(B * T, D, Hp, Wp)
            spatial_valid = spatial_layer(spatial_in[valid_flat])
            spatial_out = spatial_valid.new_zeros(B * T, D, Hp, Wp)
            spatial_out[valid_flat] = spatial_valid
            node_feat = spatial_out.view(B, T, D, Hp, Wp).permute(0, 1, 3, 4, 2)
            if return_attn:
                node_feat, attn_w = temporal_layer(node_feat, mask, return_attn=True)
                temporal_attns.append(attn_w)
            else:
                node_feat = temporal_layer(node_feat, mask)

        node_feat = self.node_norm(node_feat)

        stats = self._graph_stats(node_feat, mask)

        node_flat = node_feat.reshape(B, T, Hp * Wp, D)
        attn = torch.softmax(self.readout_pool(node_flat), dim=2)      # (B,T,N,1)
        frame_vec = (attn * node_flat).sum(dim=2)                      # (B,T,D)

        mask_f = mask.float().unsqueeze(-1)                            # (B,T,1)
        mean_vec = (frame_vec * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        first_vec = frame_vec[:, 0]
        last_idx = mask.long().sum(dim=1).sub(1).clamp(min=0)
        last_vec = frame_vec[torch.arange(B, device=x.device), last_idx]
        delta_vec = last_vec - first_vec
        graph_embed = self.temporal_readout(
            torch.cat([mean_vec, first_vec, last_vec, delta_vec], dim=1)
        )

        if return_attn:
            result = (graph_embed, stats, temporal_attns)
        else:
            result = (graph_embed, stats)
        if return_node_feat:
            # (B, T, H', W', D): learned node representations after the attention blocks and before the readout.
            result = result + (node_feat,)
        return result

    @staticmethod
    def _graph_stats(node_feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
        # the four descriptors, computed from the node features without gradient; all stay on small, stable scales
        with torch.no_grad():
            B, T, Hp, Wp, D = node_feat.shape
            flat = node_feat.reshape(B, T, Hp * Wp, D)

            # order: high when the node vectors of a frame are similar (low spatial variance = a regular arrangement)
            spatial_var = flat.var(dim=2).mean(dim=-1)                 # (B,T)
            order_curve = torch.exp(-spatial_var)                      # (B,T), (0,1]

            # disruption: mean absolute difference between each node and its horizontal and vertical neighbours, per frame
            dh = (node_feat[:, :, 1:, :, :] - node_feat[:, :, :-1, :, :]).abs().mean(dim=-1)  # (B,T,Hp-1,Wp)
            dw = (node_feat[:, :, :, 1:, :] - node_feat[:, :, :, :-1, :]).abs().mean(dim=-1)  # (B,T,Hp,Wp-1)
            disruption_curve = torch.tanh(dh.mean(dim=(2, 3)) + dw.mean(dim=(2, 3)))  # (B,T), [0,1)

            # persistence: high when node vectors change little from one frame to the next (only valid frame pairs count)
            diff_t = (flat[:, 1:] - flat[:, :-1]).abs().mean(dim=(2, 3))   # (B,T-1)
            valid_t = (mask[:, 1:] & mask[:, :-1]).float()                 # (B,T-1)
            mean_change = (diff_t * valid_t).sum(dim=1) / valid_t.sum(dim=1).clamp(min=1.0)
            persistence = torch.exp(-mean_change)                       # (B,), (0,1]

            mask_f = mask.float()
            order = (order_curve * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            disruption = (disruption_curve * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

            # recovery: positive when disruption is lower in the late half of the recording than in the early half.
            positions = torch.arange(T, device=mask.device).unsqueeze(0)
            midpoint = mask.sum(dim=1, keepdim=True).float() / 2.0
            early_mask = mask & (positions.float() < midpoint)
            late_mask = mask & ~early_mask
            early_f = early_mask.float()
            late_f = late_mask.float()
            early = (disruption_curve * early_f).sum(dim=1) / early_f.sum(dim=1).clamp(min=1.0)
            late = (disruption_curve * late_f).sum(dim=1) / late_f.sum(dim=1).clamp(min=1.0)
            recovery = torch.tanh(early - late)                         # (B,), (-1,1)

            return {
                "disruption": disruption,
                "order": order,
                "persistence": persistence,
                "recovery": recovery,
            }


class ControlGraphReference(nn.Module):
    # running mean of the descriptors of control cells, so that every cell is described as a deviation from the controls
    def __init__(self, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        for k in GRAPH_STAT_KEYS:
            self.register_buffer(f"ref_{k}", torch.zeros(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def update(self, stats: dict, is_control: torch.Tensor):
        if is_control.sum() == 0:
            return
        for k in GRAPH_STAT_KEYS:
            control_mean = stats[k][is_control].mean()
            ref = getattr(self, f"ref_{k}")
            if not bool(self.initialized.item()):
                ref.copy_(control_mean.detach().reshape(1))
            else:
                ref.copy_((1 - self.momentum) * ref + self.momentum * control_mean.detach())
        self.initialized.fill_(True)

    @torch.no_grad()
    def set_reference(self, stats_mean: dict):
        # Replace the baseline with an evaluation-mode control mean.
        for k in GRAPH_STAT_KEYS:
            value = torch.as_tensor(
                stats_mean[k], device=getattr(self, f"ref_{k}").device,
                dtype=getattr(self, f"ref_{k}").dtype,
            )
            getattr(self, f"ref_{k}").copy_(value.reshape(1))
        self.initialized.fill_(True)

    def deviation(self, stats: dict) -> dict:
        return {k: stats[k] - getattr(self, f"ref_{k}") for k in GRAPH_STAT_KEYS}


def stats_to_vector(stats: dict) -> torch.Tensor:
    # dict of (B,) tensors -> (B, len(GRAPH_STAT_KEYS)) tensor, fixed key order.
    return torch.stack([stats[k] for k in GRAPH_STAT_KEYS], dim=1)


class GraphClassifier(nn.Module):
    # encoder plus a classification head that reads the embedding and the control-referenced descriptors
    def __init__(self, num_classes: int, in_channels: int = 1, embed_dim: int = 128,
                 patch_stride: int = 8, depth: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = GraphTransitionEncoder(
            in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride,
            depth=depth, num_heads=num_heads, dropout=dropout,
        )
        self.control_ref = ControlGraphReference()
        stat_dim = len(GRAPH_STAT_KEYS)
        # Normalise the learned embedding and the four descriptors separately, so that one large descriptor cannot rescale
        # every learned feature.
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.stat_norm = nn.LayerNorm(stat_dim)
        self.head = nn.Sequential(
            nn.Identity(),  # placeholder that keeps the layer indices (head.1, head.4) of the released checkpoints
            nn.Linear(embed_dim + stat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def prepare_features(self, graph_embed, dev_stats):
        dev_vec = stats_to_vector(dev_stats)
        return torch.cat([self.embed_norm(graph_embed), self.stat_norm(dev_vec)], dim=1)

    def forward(self, x, mask, is_control=None, update_reference=False, return_attn=False, return_node_feat=False):
        encoder_out = self.encoder(x, mask, return_attn=return_attn, return_node_feat=return_node_feat)
        if return_attn and return_node_feat:
            graph_embed, stats, temporal_attns, node_feat = encoder_out
        elif return_attn:
            graph_embed, stats, temporal_attns = encoder_out
            node_feat = None
        elif return_node_feat:
            graph_embed, stats, node_feat = encoder_out
            temporal_attns = None
        else:
            graph_embed, stats = encoder_out
            temporal_attns = None
            node_feat = None

        if update_reference and is_control is not None:
            self.control_ref.update(stats, is_control)
        dev_stats = self.control_ref.deviation(stats)
        features = self.prepare_features(graph_embed, dev_stats)
        logits = self.head(features)

        if return_attn and return_node_feat:
            return logits, graph_embed, stats, dev_stats, temporal_attns, node_feat
        elif return_attn:
            return logits, graph_embed, stats, dev_stats, temporal_attns
        elif return_node_feat:
            return logits, graph_embed, stats, dev_stats, node_feat
        return logits, graph_embed, stats, dev_stats


if __name__ == "__main__":
    # Self-test with random data: shapes, a backward pass, and the check that padded frames have no effect.
    B, C, T, H, W = 3, 2, 12, 144, 144

    dummy_x = torch.randn(B, C, T, H, W)
    dummy_mask = torch.ones(B, T, dtype=torch.bool)
    dummy_mask[1, 8:] = False  # simulate a shorter sequence
    dummy_labels = torch.tensor([0, 3, 0])  # class 0 plays the role of the control class here
    is_control = dummy_labels == 0

    encoder = GraphTransitionEncoder(in_channels=C, embed_dim=32, patch_stride=8, depth=2, num_heads=4)
    graph_embed, stats = encoder(dummy_x, dummy_mask)

    print("graph_embed:", graph_embed.shape)  # (B, D)
    for k, v in stats.items():
        print(f"stats[{k}]:", v.shape, v)

    ref = ControlGraphReference()
    ref.update(stats, is_control)
    dev = ref.deviation(stats)
    print("deviation vector:", stats_to_vector(dev))

    loss = graph_embed.pow(2).mean()
    loss.backward()
    print("backward ok, loss =", loss.item())

    # Padded frames must have no effect in evaluation mode.
    encoder.eval()
    changed = dummy_x.clone()
    changed[1, :, 8:] = 1000.0 * torch.randn_like(changed[1, :, 8:])
    with torch.no_grad():
        embed_a, _ = encoder(dummy_x, dummy_mask)
        embed_b, _ = encoder(changed, dummy_mask)
    torch.testing.assert_close(embed_a[1], embed_b[1], rtol=1e-5, atol=1e-6)
    print("padding-mask invariance ok")
