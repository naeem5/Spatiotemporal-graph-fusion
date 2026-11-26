# Stage 3: train the class-wise gate on top of the two frozen experts.
# Example: python scripts/03_train_stage3_gate.py --config configs/a549_stage3_gate_seed1234.yaml --data_root /path/to/npz \
#              --app_ckpt runs/stage1_appearance_*/best_model.pth --graph_ckpt runs/stage2_graph_*/best_model.pth
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdgraph.data import CHANNELS, FusionDataset, class_weights, collate_fusion, index_recordings, read_split_table
from pdgraph.gate import AppearanceGraphGateModel, build_appearance_model, build_graph_model
from pdgraph.temporal_experts import GRAPH_ARCHITECTURES
from pdgraph.utils import build_optimizer, parse_args, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dataloaders(args, seed_worker):
    df, (train_df, val_df, test_df) = read_split_table(args.csv_path)
    # the appearance expert needs all three channels, so every recording is indexed with all of them
    train_items, val_items, test_items = [index_recordings(d, args.data_root, CHANNELS, args.min_T, desc=f"Indexing {name}")
                                          for d, name in ((train_df, "train"), (val_df, "val"), (test_df, "test"))]
    train_ds, val_ds, test_ds = [FusionDataset(items, args.graph_channels) for items in (train_items, val_items, test_items)]

    num_classes = int(df["label"].max()) + 1
    counts = np.bincount([it["label"] for it in train_items], minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"The training split has no recording of class(es) {np.where(counts == 0)[0].tolist()}")
    weight = class_weights(counts, args.weight_power)

    eval_workers = max(args.num_workers // 2, 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker, collate_fn=collate_fusion)

    def eval_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True,
                          collate_fn=collate_fusion)

    return train_loader, eval_loader(val_ds), eval_loader(test_ds), num_classes, weight


def main():
    parser = argparse.ArgumentParser(description="Stage 3: class-wise gate")
    parser.add_argument("--csv_path", type=Path, default=Path("data/split_table.csv"))
    parser.add_argument("--data_root", type=str, default="", help="folder with the npz recordings; paths in the split table are relative to it")
    parser.add_argument("--out_dir", type=Path, default=Path("runs"))
    parser.add_argument("--app_ckpt", type=Path, required=True, help="best_model.pth of Stage 1")
    parser.add_argument("--graph_ckpt", type=Path, required=True, help="best_model.pth of Stage 2")
    parser.add_argument("--graph_channels", type=str, nargs="+", default=["mito", "mmp", "nuclear"], choices=CHANNELS,
                        help="must be the channels of Stage 2, in the same order")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_T", type=int, default=3)
    parser.add_argument("--app_embed_dim", type=int, default=256)
    parser.add_argument("--graph_embed_dim", type=int, default=128)
    parser.add_argument("--graph_patch_stride", type=int, default=8)
    parser.add_argument("--graph_depth", type=int, default=2)
    parser.add_argument("--graph_num_heads", type=int, default=4)
    parser.add_argument("--graph_dropout", type=float, default=0.15, help="must match the Stage 2 checkpoint")
    parser.add_argument("--graph_arch", choices=GRAPH_ARCHITECTURES, default="graph", help="must match the Stage 2 checkpoint")
    parser.add_argument("--delta_scale", type=float, default=0.3, help="weight of the graph correction")
    parser.add_argument("--gate_tau", type=float, default=0.75, help="confidence of the appearance expert above which the prior closes the gate")
    parser.add_argument("--gate_alpha", type=float, default=10.0)
    parser.add_argument("--gate_mode", choices=["scalar", "classwise"], default="classwise")
    parser.add_argument("--gate_hidden", type=int, default=64)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--prior_beta_init", type=float, default=0.5)
    parser.add_argument("--gate_weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--weight_power", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10, help="early stopping on the selection metric")
    parser.add_argument("--selection_metric", choices=["macro_f1", "top1"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=42)
    args = parse_args(parser)

    seed_worker = set_seed(args.seed)

    run_dir = args.out_dir / ("stage3_gate_" + datetime.now().strftime(r"%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, default_flow_style=False)

    train_loader, val_loader, test_loader, num_classes, weight = get_dataloaders(args, seed_worker)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    appearance_model = build_appearance_model(args.app_ckpt, num_classes, embed_dim=args.app_embed_dim).to(device)
    graph_model = build_graph_model(args.graph_ckpt, num_classes, in_channels=len(args.graph_channels), embed_dim=args.graph_embed_dim,
                                    patch_stride=args.graph_patch_stride, depth=args.graph_depth, num_heads=args.graph_num_heads,
                                    dropout=args.graph_dropout, graph_arch=args.graph_arch).to(device)
    model = AppearanceGraphGateModel(appearance_model, graph_model, num_classes=num_classes, graph_embed_dim=args.graph_embed_dim,
                                     delta_scale=args.delta_scale, gate_tau=args.gate_tau, gate_alpha=args.gate_alpha,
                                     gate_mode=args.gate_mode, gate_hidden=args.gate_hidden, gate_dropout=args.gate_dropout,
                                     prior_beta_init=args.prior_beta_init).to(device)

    # only the gate and the strength of the confidence prior are trained; both experts stay frozen
    is_gate = lambda name: name.startswith("gate_head.") or name == "prior_beta_raw"
    trainable = [(n, p) for n, p in model.named_parameters() if is_gate(n)]
    if any(p.requires_grad and not is_gate(n) for n, p in model.named_parameters()):
        raise RuntimeError("an expert parameter is trainable; only the gate may be trained in Stage 3")
    print(f"Trainable parameters (gate only): {sum(p.numel() for _, p in trainable):,}")

    optimizer = build_optimizer(trainable, lr=args.lr, weight_decay=args.gate_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=args.label_smoothing)

    writer = SummaryWriter(run_dir)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_score, best_val_acc, best_val_f1, epochs_no_improve = -1.0, -1.0, -1.0, 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]")
        for x_app, x_graph, mask, y in pbar:
            x_app, x_graph, mask, y = x_app.to(device), x_graph.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits_final, logits_app, delta_logits, g, g_learned, g_prior = model(x_app, x_graph, mask)
                loss = criterion(logits_final, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()), gate=float(g.mean().item()))

        model.eval()
        labels, preds, preds_app, preds_graph = [], [], [], []
        gates, learned_gates, prior_gates = [], [], []
        with torch.no_grad():
            for x_app, x_graph, mask, y in tqdm(val_loader, desc="[val]"):
                x_app, x_graph, mask = x_app.to(device), x_graph.to(device), mask.to(device)
                logits_final, logits_app, delta_logits, gate, learned_gate, prior_gate = model(x_app, x_graph, mask)
                labels.append(y.numpy())
                preds.append(logits_final.argmax(dim=1).cpu().numpy())
                preds_app.append(logits_app.argmax(dim=1).cpu().numpy())
                preds_graph.append(delta_logits.argmax(dim=1).cpu().numpy())
                gates.append(gate.cpu().numpy())
                learned_gates.append(learned_gate.cpu().numpy())
                prior_gates.append(prior_gate.cpu().numpy())

        labels, preds, preds_app, preds_graph = map(np.concatenate, (labels, preds, preds_app, preds_graph))
        top1, top1_app, top1_graph = [100.0 * float((p == labels).mean()) for p in (preds, preds_app, preds_graph)]
        macro_f1, macro_f1_app, macro_f1_graph = [f1_score(labels, p, average="macro", zero_division=0) for p in (preds, preds_app, preds_graph)]
        mean_gate, mean_learned, mean_prior = [float(np.concatenate(g).mean()) for g in (gates, learned_gates, prior_gates)]
        prior_beta = float(F.softplus(model.prior_beta_raw).item())
        lr_now = optimizer.param_groups[0]["lr"]

        for name, value in (("Val/Top1", top1), ("Val/MacroF1", macro_f1), ("Val/AppearanceTop1", top1_app), ("Val/GraphTop1", top1_graph),
                            ("Val/MeanGate", mean_gate), ("Val/MeanLearnedGate", mean_learned), ("Val/MeanPriorGate", mean_prior),
                            ("Val/PriorBeta", prior_beta), ("Train/LearningRate", lr_now)):
            writer.add_scalar(name, value, epoch)
        print(f"Epoch {epoch + 1}: appearance={top1_app:.2f}%/{macro_f1_app:.4f} graph={top1_graph:.2f}%/{macro_f1_graph:.4f} "
              f"fused={top1:.2f}%/{macro_f1:.4f} gate={mean_gate:.3f} learned={mean_learned:.3f} prior={mean_prior:.3f} "
              f"beta={prior_beta:.3f} lr={lr_now:.2e}")
        print(f"  fused predicted counts: {np.bincount(preds, minlength=num_classes).tolist()}")

        torch.save(model.state_dict(), run_dir / "last_model.pth")
        score = macro_f1 if args.selection_metric == "macro_f1" else top1
        scheduler.step(score)
        if score > best_score:
            best_score, best_val_acc, best_val_f1, epochs_no_improve = score, top1, macro_f1, 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth ({args.selection_metric}={best_score:.4f}, top1={best_val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping: no improvement of val {args.selection_metric} for {args.patience} epochs. "
                      f"Best top1={best_val_acc:.2f}%, macro_f1={best_val_f1:.4f}")
                break

    writer.close()
    print(f"Training finished. Evaluate with scripts/04_evaluate.py. Checkpoint saved to: {run_dir}")


if __name__ == "__main__":
    main()
