# Stage 1: appearance expert. One frame as a 3-channel z-stack (3, 27, H, W) -> class logits.
# Parameter names match the checkpoints used in the manuscript.
import torch
import torch.nn as nn


class MultiScaleFeatureExtractor(nn.Module):
    # One token per focal plane, built from three scales of the same plane.

    def __init__(self, in_channels=3, embed_dim=256):
        super().__init__()
        # Three small convolution stages; the second and third halve the resolution, so the plane is seen at
        # full, 1/2 and 1/4 resolution.
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.stage2 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.stage3 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        # Each scale is averaged over the image and projected to the common token size, then the three are merged.
        self.fc1 = nn.Linear(32, embed_dim)
        self.fc2 = nn.Linear(64, embed_dim)
        self.fc3 = nn.Linear(128, embed_dim)
        self.fuse = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, x):
        batch, channels, planes, height, width = x.shape
        # Treat every focal plane as its own image: (B, C, Z, H, W) -> (B*Z, C, H, W).
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * planes, channels, height, width)

        f1 = self.stage1(x)     # (B*Z, 32,  H,   W)
        f2 = self.stage2(f1)    # (B*Z, 64,  H/2, W/2)
        f3 = self.stage3(f2)    # (B*Z, 128, H/4, W/4)

        # Global average over the image, one vector per scale.
        f1, f2, f3 = (f.mean(dim=[2, 3]) for f in (f1, f2, f3))

        # Project the three scales to the same size and merge them into one token per plane.
        tokens = torch.cat([self.fc1(f1), self.fc2(f2), self.fc3(f3)], dim=-1)   # (B*Z, 3D)
        tokens = self.fuse(tokens)                                               # (B*Z, D)
        return tokens.view(batch, planes, -1)                                    # (B, Z, D)


class ZTransformer(nn.Module):
    # Transformer encoder over the focal planes of one cell (a fixed number, 27, with learned positions).

    def __init__(self, embed_dim=256, depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.1, z_dim=27):
        super().__init__()
        self.z_dim = z_dim
        # Learned position of each focal plane (they are always the same 27 planes).
        self.pos_embed = nn.Parameter(torch.zeros(1, z_dim, embed_dim))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(embed_dim * mlp_ratio),
                                           dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        # tokens: (B, Z, D) from the feature extractor; returns the updated tokens, same shape.
        _, planes, _ = tokens.shape
        assert planes == self.z_dim, f"expected {self.z_dim} focal planes, got {planes}"
        x = tokens + self.pos_embed
        x = self.encoder(x)
        return self.norm(x)


class AppearanceExpert(nn.Module):
    # Stage 1: classify a cell from the z-stack of its middle frame.

    def __init__(self, num_classes=5, embed_dim=256):
        super().__init__()
        self.extractor = MultiScaleFeatureExtractor(in_channels=3, embed_dim=embed_dim)
        self.transformer = ZTransformer(embed_dim=embed_dim, z_dim=27)
        # Attention pooling: a learned score per focal plane decides how much each plane contributes.
        self.attn_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def encode(self, x):
        # pooled cell representation (B, D) and class logits
        tokens = self.extractor(x)                      # (B, 27, D)
        planes = self.transformer(tokens)               # (B, 27, D)
        weights = torch.softmax(self.attn_pool(planes), dim=1)   # (B, 27, 1), sums to 1 over the planes
        pooled = (weights * planes).sum(dim=1)          # (B, D)
        return pooled, self.fc(pooled)

    def forward(self, x):
        return self.encode(x)[1]
