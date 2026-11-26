# Stage 1, the appearance expert (2D variant: one frame, one focal plane), on a public data set.
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.adam import Adam
from tqdm import tqdm

import config as cfg
import data_loading
import models
import run_utils


def evaluate_frames(net, loader, device):
    net.eval()
    labels, preds, pmax = [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="[evaluate]", leave=False):
            logits = net(x.to(device))
            labels.append(y.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            pmax.append(torch.softmax(logits, 1).max(1).values.cpu().numpy())
    labels, preds = np.concatenate(labels), np.concatenate(preds)
    top1, macro_f1 = run_utils.top1_and_macro_f1(labels, preds)
    return top1, macro_f1, labels, preds, np.concatenate(pmax)


def main():
    s = cfg.STAGE1
    if cfg.SKIP_FINISHED and run_utils.registered_run("stage1") is not None:
        print(f"Stage 1 is already registered ({run_utils.registered_run('stage1')}); set SKIP_FINISHED = False to train again.")
        return
    device = models.device()
    data_loading.set_seed(cfg.STAGE1_SEED)
    run_dir = run_utils.new_run_dir("stage1_appearance", __file__)
    print(f"[Stage 1] {cfg.RUN_TAG}; device {device}; run folder {run_dir}")

    train_rec, val_rec, test_rec = (data_loading.load_index(sp, min_T=1) for sp in ("train", "val", "test"))
    weight = data_loading.class_weights(train_rec, s["weight_power"]).to(device)
    half = max(cfg.NUM_WORKERS // 2, 0)
    train_loader = data_loading.make_loader(data_loading.AnchorFrameDataset(train_rec, augment=s.get("augment", False)), s["batch_size"], True, True, seed=cfg.STAGE1_SEED)
    val_loader = data_loading.make_loader(data_loading.AnchorFrameDataset(val_rec), s["batch_size"], False, False, workers=half)
    test_loader = data_loading.make_loader(data_loading.AnchorFrameDataset(test_rec), s["batch_size"], False, False, workers=half)
    print(f"sequences: train {len(train_rec)}, val {len(val_rec)}, test {len(test_rec)}")

    net = models.build_stage1().to(device)
    optimizer = Adam(net.parameters(), lr=s["base_lr"] * s["batch_size"] / 256, weight_decay=s["weight_decay"])
    schedule = run_utils.WarmupCosineSchedule(optimizer, s["warmup_epochs"], s["epochs"], s["min_lr"])
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=s["label_smoothing"])
    scaler = GradScaler(enabled=torch.cuda.is_available())
    writer = run_utils.make_writer(run_dir)

    best_top1 = -1.0
    for epoch in range(s["epochs"]):
        started = time.time()
        schedule.step(epoch)
        net.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{s['epochs']} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                loss = criterion(net(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        top1, macro_f1, *_ = evaluate_frames(net, val_loader, device)
        writer.add_scalar("Val/Top1", top1, epoch)
        print(f"Epoch {epoch + 1}: val top-1 {top1:.2f}%  macro-F1 {macro_f1:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  {(time.time() - started) / 60:.1f} min")
        torch.save(net.state_dict(), run_dir / "last_model.pth")
        if top1 > best_top1:
            best_top1 = top1
            torch.save(net.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth (top-1 {best_top1:.2f}%)")
    writer.close()

    net.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    rows = []
    for name, loader in (("val", val_loader), ("test", test_loader)):
        top1, macro_f1, labels, preds, pmax = evaluate_frames(net, loader, device)
        print(f"[{name}] top-1 {top1:.2f}%  macro-F1 {macro_f1:.4f}")
        rows += [dict(split=name, label=int(l), pred=int(p), pmax=float(m)) for l, p, m in zip(labels, preds, pmax)]
    pd.DataFrame(rows).to_csv(run_dir / "stage1_export.csv", index=False)
    run_utils.register_run("stage1", run_dir)
    print(f"Done. Stage 1 saved to {run_dir}")


if __name__ == "__main__":
    main()
