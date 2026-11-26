import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim import AdamW


def get_device(index=0):
    # GPU number `index` when available, otherwise the CPU
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class WarmupCosineSchedule:
    # linear warm-up from min_lr to the base learning rate, then cosine decay back to min_lr; call step(epoch) once per epoch
    def __init__(self, optimizer, warmup_epochs, epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.min_lr = min_lr
        self.base_lr = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch):
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lr):
            if epoch < self.warmup_epochs:
                lr = self.min_lr + (base_lr - self.min_lr) * epoch / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            group["lr"] = lr


def build_optimizer(named_parameters, lr, weight_decay):
    # AdamW; weight decay only on weight matrices and kernels, not on biases and normalisation parameters
    decay, no_decay = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def parse_args(parser):
    # --config <file.yaml> supplies the defaults; arguments given on the command line override it
    parser.add_argument("--config", type=Path, default=None, help="YAML file with default values for the arguments")
    known, _ = parser.parse_known_args()
    if known.config is not None:
        parser.set_defaults(**yaml.safe_load(known.config.read_text(encoding="utf-8")))
    return parser.parse_args()
