# the three-stage model for the public datasets: Stage 1 is a 2D ResNet-18 (these datasets have no z-stack); Stage 2 and the gate come from pdgraph
import torch.nn as nn

import config as cfg
from pdgraph.gate import assemble_gate_model
from pdgraph.temporal_experts import graph_classifier_class
from pdgraph.utils import get_device


def device():
    return get_device(cfg.DEVICE_INDEX)


class ResNetAppearance(nn.Module):
    # Stage 1 for data without a z-stack: torchvision ResNet-18 (from scratch, first layer for C channels) and a linear head. Input (B, C, 1, H, W).

    def __init__(self, num_classes, in_channels):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(weights=None)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def encode(self, x):
        pooled = self.backbone(x[:, :, 0])
        return pooled, self.fc(pooled)

    def forward(self, x):
        return self.encode(x)[1]


def build_stage1():
    return ResNetAppearance(cfg.num_classes(), len(cfg.CHANNELS))


def build_stage2():
    s = cfg.STAGE2
    return graph_classifier_class(cfg.STAGE2_ARCH)(num_classes=cfg.num_classes(), in_channels=len(cfg.CHANNELS), embed_dim=s["embed_dim"],
                                                   patch_stride=s["patch_stride"], depth=s["depth"], num_heads=s["num_heads"], dropout=s["dropout"])


def build_stage3(app_ckpt=None, graph_ckpt=None, delta_scale=None):
    # gate model with frozen experts, optionally loaded from the Stage 1 / Stage 2 checkpoints
    return assemble_gate_model(build_stage1(), build_stage2(), cfg.num_classes(), cfg.STAGE2["embed_dim"], cfg.STAGE3, app_ckpt, graph_ckpt, delta_scale)
