# Stage 2: train the graph expert on the maximum projection of all frames of each recording.
# Example: python scripts/02_train_stage2_graph.py --config configs/a549_stage2_graph.yaml --data_root /path/to/npz
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdgraph.data import GraphSequenceDataset, class_weights, collate_graph, index_recordings, read_split_table
from pdgraph.graph_expert import GRAPH_STAT_KEYS
from pdgraph.temporal_experts import GRAPH_ARCHITECTURES, graph_classifier_class
from pdgraph.utils import WarmupCosineSchedule, build_optimizer, parse_args, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_control_label(csv_path, control_class):
    meta = pd.read_csv(csv_path)
    matches = meta[meta["class"].astype(str).str.strip().str.lower() == control_class.lower()]
    if matches.empty:
        raise ValueError(f"No rows with class == '{control_class}' in {csv_path}")
    return int(matches["label"].iloc[0])


def get_dataloaders(args, seed_worker, control_label):
    df, (train_df, val_df, test_df) = read_split_table(args.csv_path)
    train_items, val_items, test_items = [index_recordings(d, args.data_root, args.channels, args.min_T, desc=f"Indexing {name}")
                                          for d, name in ((train_df, "train"), (val_df, "val"), (test_df, "test"))]
    train_ds, val_ds, test_ds = [GraphSequenceDataset(items, args.channels) for items in (train_items, val_items, test_items)]

    num_classes = int(df["label"].max()) + 1
    counts = np.bincount([it["label"] for it in train_items], minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"The training split has no recording of class(es) {np.where(counts == 0)[0].tolist()}")
    weight = class_weights(counts, args.weight_power)

    # control recordings of the training split, used to measure the control baseline of the descriptors
    reference_ds = GraphSequenceDataset([it for it in train_items if it["label"] == control_label], args.channels)

    eval_workers = max(args.num_workers // 2, 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
                              collate_fn=collate_graph)

    def eval_loader(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True, collate_fn=collate_graph)

    return train_loader, eval_loader(val_ds), eval_loader(test_ds), eval_loader(reference_ds), eval_loader(train_ds), num_classes, weight


@torch.no_grad()
def calibrate_control_reference(model, reference_loader):
    # control baseline of the descriptors, measured in eval mode (dropout off, BatchNorm with running statistics)
    model.eval()
    sums = {k: torch.zeros((), dtype=torch.float64, device=device) for k in GRAPH_STAT_KEYS}
    count = 0
    for x, mask, lengths, y in tqdm(reference_loader, desc="[control calibration]", leave=False):
        x, mask = x.to(device), mask.to(device)
        _, stats = model.encoder(x, mask)
        for k in GRAPH_STAT_KEYS:
            sums[k] += stats[k].double().sum()
        count += x.size(0)
    if count == 0:
        raise RuntimeError("No control recordings available to calibrate the control reference")
    means = {k: (sums[k] / count).float() for k in GRAPH_STAT_KEYS}
    model.control_ref.set_reference(means)
    return {k: float(v.item()) for k, v in means.items()}


def main():
    parser = argparse.ArgumentParser(description="Stage 2: graph expert")
    parser.add_argument("--csv_path", type=Path, default=Path("data/split_table.csv"))
    parser.add_argument("--data_root", type=str, default="", help="folder with the npz recordings; paths in the split table are relative to it")
    parser.add_argument("--out_dir", type=Path, default=Path("runs"))
    parser.add_argument("--channels", type=str, nargs="+", default=["mito", "mmp", "nuclear"], choices=["mito", "mmp", "nuclear"],
                        help="graph channels; Stage 3 must use the same order")
    parser.add_argument("--control_class", type=str, default="Control", help="class that defines the control baseline")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--min_T", type=int, default=3, help="recordings with fewer frames are not used")
    parser.add_argument("--graph_arch", choices=GRAPH_ARCHITECTURES, default="graph", help="graph = patch graph; temporal, temporal_nodes = control experts without spatial edges")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_power", type=float, default=0.5, help="class weights: inverse frequency to this power")
    parser.add_argument("--patience", type=int, default=15, help="early stopping on the selection metric")
    parser.add_argument("--selection_metric", choices=["macro_f1", "top1"], default="macro_f1")
    parser.add_argument("--seed", type=int, default=1234)
    args = parse_args(parser)

    seed_worker = set_seed(args.seed)
    control_label = find_control_label(args.csv_path, args.control_class)
    print(f"[Info] control label id = {control_label}")

    run_dir = args.out_dir / ("stage2_graph_" + datetime.now().strftime(r"%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, default_flow_style=False)

    train_loader, val_loader, test_loader, reference_loader, export_train_loader, num_classes, weight = get_dataloaders(args, seed_worker, control_label)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model = graph_classifier_class(args.graph_arch)(num_classes=num_classes, in_channels=len(args.channels), embed_dim=args.embed_dim,
                            patch_stride=args.patch_stride, depth=args.depth, num_heads=args.num_heads, dropout=args.dropout).to(device)

    epochs = args.epochs
    lr = 1e-3 * args.batch_size / 64             # base learning rate scaled with the batch size
    optimizer = build_optimizer(model.named_parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_epochs=max(5, epochs // 10), epochs=epochs, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.1)

    writer = SummaryWriter(run_dir)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_score, best_val_acc, best_val_f1, best_val_top5, best_epoch, epochs_no_improve = -1.0, -1.0, -1.0, 0.0, 0, 0
    for epoch in range(epochs):
        scheduler.step(epoch)
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]")
        train_loss_sum, train_correct, train_n = 0.0, 0, 0

        for x, mask, lengths, y in pbar:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            is_control = y == control_label

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                # the running control baseline of the descriptors is updated with every training batch
                logits, graph_embed, stats, dev_stats = model(x, mask, is_control=is_control, update_reference=True)
                loss = criterion(logits, y)

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * y.size(0)
            train_correct += int((logits.argmax(dim=1) == y).sum().item())
            train_n += y.size(0)
            pbar.set_postfix(loss=f"{train_loss_sum / max(train_n, 1):.4f}", acc=f"{100.0 * train_correct / max(train_n, 1):.2f}")

        # the baseline used for validation is measured in eval mode
        reference_means = calibrate_control_reference(model, reference_loader)

        model.eval()
        correct, correct5, n_val = 0.0, 0.0, 0
        all_preds, all_labels = [], []
        topk = min(5, num_classes)
        with torch.no_grad():
            for x, mask, lengths, y in tqdm(val_loader, desc="[val]"):
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits, *_ = model(x, mask)
                correct += (logits.argmax(dim=1) == y).float().sum().item()
                correct5 += (logits.topk(topk, dim=1)[1] == y.unsqueeze(1)).any(dim=1).float().sum().item()
                n_val += y.size(0)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_labels.append(y.cpu().numpy())

        top1, top5 = 100.0 * correct / max(n_val, 1), 100.0 * correct5 / max(n_val, 1)
        macro_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro", zero_division=0)
        train_loss, train_acc = train_loss_sum / max(train_n, 1), 100.0 * train_correct / max(train_n, 1)
        pred_counts = np.bincount(np.concatenate(all_preds), minlength=num_classes)     # shows a collapse onto one class
        current_lr = optimizer.param_groups[0]["lr"]
        for name, value in (("Train/Loss", train_loss), ("Train/Top1", train_acc), ("Train/LR", current_lr),
                            ("Val/Top1", top1), ("Val/Top5", top5), ("Val/MacroF1", macro_f1)):
            writer.add_scalar(name, value, epoch)
        print(f"Epoch {epoch + 1}: train loss={train_loss:.4f} top1={train_acc:.2f}% | "
              f"val top1={top1:.2f}% top5={top5:.2f}% macro_f1={macro_f1:.4f} lr={current_lr:.3e}")
        print(f"  val predicted counts: {pred_counts.tolist()}")
        print(f"  control reference: {reference_means}")

        torch.save(model.state_dict(), run_dir / "last_model.pth")
        score = macro_f1 if args.selection_metric == "macro_f1" else top1
        if score > best_score:
            best_score, best_val_acc, best_val_f1, best_val_top5, best_epoch, epochs_no_improve = score, top1, macro_f1, top5, epoch + 1, 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth ({args.selection_metric}={best_score:.4f}, top1={best_val_acc:.2f}%, top5={top5:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}. Best epoch {best_epoch}: top1={best_val_acc:.2f}%, "
                      f"top5={best_val_top5:.2f}%, macro_f1={best_val_f1:.4f}")
                break

    writer.close()
    print(f"Training finished. Best epoch {best_epoch}: top1={best_val_acc:.2f}%, top5={best_val_top5:.2f}%, macro_f1={best_val_f1:.4f}")

    # descriptors and predictions of the best checkpoint for every recording (train, val, test)
    print("Exporting graph statistics...")
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    model.eval()

    rows = []
    test_correct1, test_correct5, test_n = 0.0, 0.0, 0
    with torch.no_grad():
        for loader, split_name in [(export_train_loader, "train"), (val_loader, "val"), (test_loader, "test")]:
            for x, mask, lengths, y in tqdm(loader, desc=f"[export {split_name}]"):
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                logits, graph_embed, stats, dev_stats = model(x, mask)
                pred = logits.argmax(dim=1)
                if split_name == "test":
                    test_correct1 += (pred == y).float().sum().item()
                    test_correct5 += (logits.topk(topk, dim=1)[1] == y.unsqueeze(1)).any(dim=1).float().sum().item()
                    test_n += y.size(0)
                for i in range(x.size(0)):
                    row = {"split": split_name, "label": int(y[i].item()), "pred": int(pred[i].item()), "T": int(lengths[i].item())}
                    for k in GRAPH_STAT_KEYS:
                        row[k] = float(stats[k][i].item())
                        row[f"{k}_dev"] = float(dev_stats[k][i].item())
                    rows.append(row)

    print(f"Test (best checkpoint): top-1 = {100.0 * test_correct1 / max(test_n, 1):.2f}%, top-5 = {100.0 * test_correct5 / max(test_n, 1):.2f}%")
    pd.DataFrame(rows).to_csv(run_dir / "stage2_export.csv", index=False)
    print(f"Done. Checkpoint and exports saved to: {run_dir}")


if __name__ == "__main__":
    main()
