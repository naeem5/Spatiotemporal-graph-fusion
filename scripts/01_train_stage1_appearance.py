# Stage 1: train the appearance expert on single frames (raw 3-channel z-stacks).
# Example: python scripts/01_train_stage1_appearance.py --config configs/a549_stage1_appearance.yaml --data_root /path/to/npz
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
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdgraph.appearance import AppearanceExpert
from pdgraph.data import AppearanceFrameDataset, class_weights, index_recordings, read_split_table
from pdgraph.utils import WarmupCosineSchedule, parse_args, set_seed


def get_dataloaders(args, seed_worker):
    _, (train_df, val_df, test_df) = read_split_table(args.csv_path)
    train_ds, val_ds, test_ds = [AppearanceFrameDataset(index_recordings(d, args.data_root, desc=f"Indexing {name}"))
                                 for d, name in ((train_df, "train"), (val_df, "val"), (test_df, "test"))]

    num_classes = int(train_df["label"].nunique())
    weight = class_weights(train_df["label"].value_counts().sort_index().values, args.weight_power)

    eval_workers = max(args.num_workers // 2, 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=eval_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, num_classes, weight


def main():
    parser = argparse.ArgumentParser(description="Stage 1: appearance expert")
    parser.add_argument("--csv_path", type=Path, default=Path("data/split_table.csv"))
    parser.add_argument("--data_root", type=str, default="", help="folder with the npz recordings; paths in the split table are relative to it")
    parser.add_argument("--out_dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--weight_power", type=float, default=1.0)
    parser.add_argument("--lr_schedule_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)
    args = parse_args(parser)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] using device: {device}")
    seed_worker = set_seed(args.seed)

    run_dir = args.out_dir / ("stage1_appearance_" + datetime.now().strftime(r"%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, run_dir)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, default_flow_style=False)

    train_loader, val_loader, test_loader, num_classes, weight = get_dataloaders(args, seed_worker)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    net = AppearanceExpert(num_classes=num_classes, embed_dim=args.embed_dim).to(device)

    epochs = args.epochs
    lr_schedule_epochs = args.lr_schedule_epochs or epochs
    lr = 3e-3 * args.batch_size / 256            # base learning rate scaled with the batch size
    optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = WarmupCosineSchedule(optimizer, warmup_epochs=5, epochs=lr_schedule_epochs, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss(weight=weight.to(device), label_smoothing=0.1)

    writer = SummaryWriter(run_dir)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = -1.0
    for epoch in range(epochs):
        scheduler.step(epoch)
        net.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                loss = criterion(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=float(loss.item()))

        net.eval()
        n_correct, n_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="[val]"):
                x, y = x.to(device), y.to(device)
                pred = net(x).argmax(dim=1)
                n_correct += int((pred == y).sum())
                n_total += y.size(0)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        top1 = 100.0 * n_correct / max(n_total, 1)
        macro_f1 = f1_score(np.concatenate(all_labels), np.concatenate(all_preds), average="macro", zero_division=0)
        writer.add_scalar("Val/Top1", top1, epoch)
        writer.add_scalar("Val/MacroF1", macro_f1, epoch)
        print(f"Epoch {epoch + 1}: val top1={top1:.2f}%  macro_f1={macro_f1:.4f}")

        torch.save(net.state_dict(), run_dir / "last_model.pth")
        if top1 > best_val_acc:               # the best epoch is chosen by validation top-1 accuracy
            best_val_acc = top1
            torch.save(net.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth (top1={best_val_acc:.2f}%)")

    writer.close()

    # predictions of the best checkpoint for every sample
    print("Exporting predictions...")
    net.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    net.eval()
    rows = []
    with torch.no_grad():
        for loader, split_name in [(train_loader, "train"), (val_loader, "val"), (test_loader, "test")]:
            for x, y in tqdm(loader, desc=f"[export {split_name}]"):
                logits = net(x.to(device))
                probs = torch.softmax(logits, dim=1)
                top5 = logits.topk(min(5, num_classes), dim=1).indices.cpu()
                for i in range(x.size(0)):
                    label_i = int(y[i].item())
                    rows.append({"split": split_name, "label": label_i, "pred": int(logits[i].argmax().item()),
                                 "pmax": float(probs[i].max().item()), "top5_correct": bool(label_i in top5[i].tolist())})
    pd.DataFrame(rows).to_csv(run_dir / "stage1_export.csv", index=False)
    print(f"Done. Checkpoint and exports saved to: {run_dir}")


if __name__ == "__main__":
    main()
