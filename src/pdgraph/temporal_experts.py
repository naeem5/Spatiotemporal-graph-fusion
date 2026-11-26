import torch
import torch.nn as nn

from pdgraph.graph_expert import GraphClassifier, GraphTransitionEncoder, PatchFeatureMap, TemporalNodeAttention

GRAPH_ARCHITECTURES = ("graph", "temporal", "temporal_nodes")


# Control experts for the patch graph: same per-frame CNN, temporal attention, readout and head, but no spatial attention.
class TemporalOnlyEncoder(nn.Module):
    # pool_grid=True: the node grid of each frame is averaged to one vector, so no spatial arrangement is left.
    # pool_grid=False: the grid is kept and every node attends across time only.
    def __init__(self, in_channels=1, embed_dim=128, patch_stride=8, depth=2, num_heads=4, dropout=0.1, pool_grid=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_grid = pool_grid
        self.backbone = PatchFeatureMap(in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride)
        self.temporal_layers = nn.ModuleList([TemporalNodeAttention(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)])
        self.node_norm = nn.LayerNorm(embed_dim)
        self.readout_pool = nn.Linear(embed_dim, 1)
        self.temporal_readout = nn.Sequential(nn.LayerNorm(embed_dim * 4), nn.Linear(embed_dim * 4, embed_dim), nn.GELU())

    def forward(self, x, mask, return_attn=False, return_node_feat=False):
        if return_attn or return_node_feat:
            raise NotImplementedError("the temporal-only experts return neither attention maps nor node features")
        batch, channels, frames, height, width = x.shape
        if mask.shape != (batch, frames):
            raise ValueError(f"mask must have shape {(batch, frames)}, got {tuple(mask.shape)}")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid frame")

        # the CNN only sees real frames, so zero padding does not enter the batch statistics
        flat_frames = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        valid = mask.reshape(batch * frames)
        valid_features = self.backbone(flat_frames[valid])
        _, dim, grid_h, grid_w = valid_features.shape
        features = valid_features.new_zeros(batch * frames, dim, grid_h, grid_w)
        features[valid] = valid_features
        node_feat = features.view(batch, frames, dim, grid_h, grid_w).permute(0, 1, 3, 4, 2)        # (B, T, H', W', D)
        if self.pool_grid:
            node_feat = node_feat.mean(dim=(2, 3), keepdim=True)                                   # (B, T, 1, 1, D)

        position = GraphTransitionEncoder._temporal_position_encoding(frames, dim, x.device, node_feat.dtype)
        node_feat = node_feat + position[None, :, None, None, :] * mask[:, :, None, None, None].to(node_feat.dtype)
        for layer in self.temporal_layers:
            node_feat = layer(node_feat, mask)
        node_feat = self.node_norm(node_feat)
        stats = self._pooled_stats(node_feat, mask) if self.pool_grid else GraphTransitionEncoder._graph_stats(node_feat, mask)

        n_frames, n_nodes = node_feat.shape[1], node_feat.shape[2] * node_feat.shape[3]
        nodes = node_feat.reshape(batch, n_frames, n_nodes, dim)
        weights = torch.softmax(self.readout_pool(nodes), dim=2)
        frame_vec = (weights * nodes).sum(dim=2)                                                   # (B, T, D)
        valid_f = mask.float().unsqueeze(-1)
        mean_vec = (frame_vec * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1.0)
        first_vec = frame_vec[:, 0]
        last_index = mask.long().sum(dim=1).sub(1).clamp(min=0)
        last_vec = frame_vec[torch.arange(batch, device=x.device), last_index]
        embedding = self.temporal_readout(torch.cat([mean_vec, first_vec, last_vec, last_vec - first_vec], dim=1))
        return embedding, stats

    @staticmethod
    def _pooled_stats(node_feat, mask):
        # persistence as in GraphTransitionEncoder._graph_stats; disruption, order and recovery need the spatial grid and are set to 0
        with torch.no_grad():
            batch, frames = node_feat.shape[:2]
            flat = node_feat.reshape(batch, frames, -1, node_feat.shape[-1])
            change = (flat[:, 1:] - flat[:, :-1]).abs().mean(dim=(2, 3))
            valid_pairs = (mask[:, 1:] & mask[:, :-1]).float()
            persistence = torch.exp(-(change * valid_pairs).sum(dim=1) / valid_pairs.sum(dim=1).clamp(min=1.0))
            zeros = torch.zeros_like(persistence)
            return {"disruption": zeros, "order": zeros.clone(), "persistence": persistence, "recovery": zeros.clone()}


class TemporalClassifier(GraphClassifier):
    # GraphClassifier with the temporal-only encoder; control reference, normalisation layers and head are unchanged
    def __init__(self, num_classes, in_channels=1, embed_dim=128, patch_stride=8, depth=2, num_heads=4, dropout=0.1, pool_grid=True):
        super().__init__(num_classes=num_classes, in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride, depth=depth,
                         num_heads=num_heads, dropout=dropout)
        self.encoder = TemporalOnlyEncoder(in_channels=in_channels, embed_dim=embed_dim, patch_stride=patch_stride, depth=depth,
                                           num_heads=num_heads, dropout=dropout, pool_grid=pool_grid)


def graph_classifier_class(architecture):
    # class to construct for a Stage 2 architecture; accepts the same keyword arguments as GraphClassifier
    if architecture == "graph":
        return GraphClassifier
    if architecture not in GRAPH_ARCHITECTURES:
        raise ValueError(f"architecture must be one of {GRAPH_ARCHITECTURES}, not {architecture!r}")

    class _Classifier(TemporalClassifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, pool_grid=(architecture == "temporal"), **kwargs)

    _Classifier.__name__ = _Classifier.__qualname__ = f"TemporalClassifier_{architecture}"
    return _Classifier


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
