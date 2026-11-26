# Train the 3D ResNet-34 comparison network on the split table and score it on the test recordings.

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as bm
from pdgraph.data import AppearanceFrameDataset, class_weights, index_recordings, load_array, pad_hw, read_split_table
from pdgraph.utils import WarmupCosineSchedule, set_seed

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA else "cpu")
CHANNELS = ["mito", "mmp", "nuclear"]
TMAX = 9


class AllFramesDataset(Dataset):
    # every frame of a recording as a (3, Z, H, W) z-stack, for scoring the 3D ResNet
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        vols = {c: np.asarray(load_array(item["paths"][c])) for c in CHANNELS}
        frames = [pad_hw(np.stack([vols[c][t] for c in CHANNELS]).astype(np.float32)) for t in range(item["T"])]
        return np.ascontiguousarray(np.stack(frames)), item["label"], item["T"]


def all_frames_collate(batch):
    xs, ys, ts = zip(*batch)
    x = torch.zeros((len(xs), max(ts)) + xs[0].shape[1:], dtype=torch.float32)
    mask = torch.zeros((len(xs), max(ts)), dtype=torch.bool)
    for i, a in enumerate(xs):
        x[i, : a.shape[0]] = torch.from_numpy(a)
        mask[i, : a.shape[0]] = True
    return x, mask, torch.tensor(ys, dtype=torch.long), torch.tensor(ts, dtype=torch.long)


def build_model():
    model = bm.ResNet3D(bm.BasicBlock3D, [3, 4, 6, 3], num_classes=10, in_channels=3)
    return bm.set_bn_eps(model, 1e-4).to(DEVICE)


def forward(model, batch):
    x, y = batch
    return model(x.to(DEVICE, non_blocking=True)), y.to(DEVICE)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="[val]", leave=False):
        logits, y = forward(model, batch)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    f1 = []
    for c in range(10):
        tp, fp, fn = ((preds == c) & (labels == c)).sum(), ((preds == c) & (labels != c)).sum(), ((preds != c) & (labels == c)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float((preds == labels).mean() * 100), float(np.mean(f1))


@torch.no_grad()
def score_test(model, test_items, workers):
    # every frame is classified on its own; a recording is scored on its middle frame (and on the mean of its frame logits)
    model.eval()
    loader = DataLoader(AllFramesDataset(test_items), batch_size=4, shuffle=False, num_workers=workers, collate_fn=all_frames_collate)
    frame_logits, labels, lengths = [], [], []
    for x, mask, y, t in tqdm(loader, desc="[test]"):
        out = torch.cat([model(chunk.to(DEVICE)).float().cpu() for chunk in x[mask].split(8)])
        fl = torch.full((x.shape[0], TMAX, out.shape[1]), float("nan"))
        fl[:, : mask.shape[1]][mask] = out
        frame_logits.append(fl.numpy())
        labels.append(y.numpy())
        lengths.append(t.numpy())
    frame_logits, lengths = np.concatenate(frame_logits), np.concatenate(lengths)
    return dict(logits_mid=frame_logits[np.arange(len(lengths)), lengths // 2], logits_mean=np.nanmean(frame_logits, axis=1),
                labels=np.concatenate(labels), T=lengths)


def train_one(args, seed, data, test_items):
    out = args.out_dir / f"baseline_r3d_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size
    train_ds, val_ds, weight = data
    seed_worker = set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=CUDA, drop_last=True,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(args.num_workers // 2, 0), pin_memory=CUDA)

    model = build_model()
    lr = 3e-3 * batch_size / 256
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    schedule = WarmupCosineSchedule(optimizer, 5, args.epochs, 1e-6)
    criterion = nn.CrossEntropyLoss(weight=weight.to(DEVICE), label_smoothing=0.1)
    scaler = GradScaler(enabled=CUDA)
    (out / "config.yaml").write_text(yaml.dump(dict(model="3D ResNet-34", seed=seed, epochs=args.epochs, batch_size=batch_size, lr=lr, train=len(train_ds), val=len(val_ds))))

    state_file, start, best = out / "state.pt", 0, -1.0
    if state_file.exists():                                  # an interrupted run continues from its last epoch
        s = torch.load(state_file, map_location="cpu", weights_only=False)
        model.load_state_dict(s["model"])
        optimizer.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"])
        start, best = int(s["epoch"]) + 1, float(s["best"])
    for epoch in range(start, args.epochs):
        started = time.time()
        model.train()
        total, n, correct = 0.0, 0, 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]", leave=False)):
            schedule.step(epoch + i / len(train_loader))
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=CUDA):
                logits, y = forward(model, batch)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss) * y.size(0)
            n += y.size(0)
            correct += int((logits.argmax(1) == y).sum())
        top1, f1 = validate(model, val_loader)
        print(f"epoch {epoch + 1}: loss {total / n:.4f} train top-1 {100 * correct / n:.2f}% | val top-1 {top1:.2f}% macro-F1 {f1:.4f} ({(time.time() - started) / 60:.1f} min)")
        if top1 > best:                                       # the best epoch is chosen by validation top-1
            best = top1
            torch.save(model.state_dict(), out / "best_model.pth")
        torch.save(dict(model=model.state_dict(), opt=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best=best), state_file)

    model.load_state_dict(torch.load(out / "best_model.pth", map_location=DEVICE, weights_only=True))
    result = score_test(model, test_items, args.num_workers)
    result["graph_path"] = np.array([it["rel_path"] for it in test_items], dtype=object)
    np.savez_compressed(args.out_dir / f"baseline_r3d_seed{seed}.npz", **result)
    print(f"saved {args.out_dir / f'baseline_r3d_seed{seed}.npz'}")


def main():
    parser = argparse.ArgumentParser(description="comparison networks on the split table")
    parser.add_argument("--csv_path", type=Path, default=Path("data/split_table.csv"))
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/baselines"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 42, 2025])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, (train_df, val_df, test_df) = read_split_table(args.csv_path)
    train_items = index_recordings(train_df, args.data_root, CHANNELS, 1, desc="Indexing train")
    val_items = index_recordings(val_df, args.data_root, CHANNELS, 1, desc="Indexing val")
    test_items = index_recordings(test_df, args.data_root, CHANNELS, 3, desc="Indexing test")        # the test recordings are those of the other experiments
    weight = class_weights(np.bincount([it["label"] for it in train_items], minlength=10), 1.0)
    train_ds, val_ds = AppearanceFrameDataset(train_items), AppearanceFrameDataset(val_items)
    for seed in args.seeds:
        train_one(args, seed, (train_ds, val_ds, weight), test_items)


if __name__ == "__main__":
    main()
