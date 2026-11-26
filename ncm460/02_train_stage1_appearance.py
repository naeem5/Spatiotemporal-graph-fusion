# Stage 1, the appearance expert, trained on NCM460.
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
    # Top-1 (%), macro-F1, and per-frame predictions of a loader of single frames.
    net.eval()
    labels, preds, pmax, top5 = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="[evaluate]", leave=False):
            logits = net(x.to(device))
            prob = torch.softmax(logits, dim=1)
            labels.append(y.numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            pmax.append(prob.max(1).values.cpu().numpy())
            top5.append((logits.topk(min(5, cfg.NUM_CLASSES), dim=1).indices.cpu() == y.unsqueeze(1)).any(1).numpy())
    labels, preds = np.concatenate(labels), np.concatenate(preds)
    top1, macro_f1 = run_utils.top1_and_macro_f1(labels, preds)
    return top1, macro_f1, labels, preds, np.concatenate(pmax), np.concatenate(top5)


def make_loaders(train_rec, val_rec, test_rec):
    # Training, validation and test loaders of single frames in the reading mode of the configuration.
    s, half = cfg.STAGE1, max(cfg.NUM_WORKERS // 2, 0)
    if cfg.STAGE1_READ_MODE == "middle_cache":
        for split, recs in (("train", train_rec), ("val", val_rec)):           # the cache must belong to the current index
            cached = np.load(cfg.STAGE1_CACHE_DIR / f"{split}_labels.npy")
            if len(cached) != len(recs) or not np.array_equal(cached, [int(r["label"]) for r in recs]):
                raise ValueError(f"the {split} cache in {cfg.STAGE1_CACHE_DIR} does not match the index; run 01f_cache_stage1_frames.py")
        train = data_loading.make_loader(data_loading.CachedFrameDataset("train"), s["batch_size"], True, True, seed=cfg.STAGE1_SEED)
        val = data_loading.make_loader(data_loading.CachedFrameDataset("val"), s["batch_size"], False, False, workers=half)
        test = data_loading.make_loader(data_loading.MiddleFrameDataset(test_rec), s["batch_size"], False, False, workers=half)
    elif cfg.STAGE1_READ_MODE == "stream":
        train = data_loading.make_stream_loader(data_loading.FrameStreamDataset(train_rec, True, cfg.STAGE1_SEED), s["batch_size"], drop_last=True)
        val = data_loading.make_stream_loader(data_loading.FrameStreamDataset(val_rec, False), s["batch_size"], drop_last=False, workers=half)
        test = data_loading.make_stream_loader(data_loading.FrameStreamDataset(test_rec, False), s["batch_size"], drop_last=False, workers=half)
    elif cfg.STAGE1_READ_MODE == "frame":
        train = data_loading.make_loader(data_loading.AppearanceFrameDataset(train_rec), s["batch_size"], True, True, seed=cfg.STAGE1_SEED)
        val = data_loading.make_loader(data_loading.AppearanceFrameDataset(val_rec), s["batch_size"], False, False, workers=half)
        test = data_loading.make_loader(data_loading.AppearanceFrameDataset(test_rec), s["batch_size"], False, False, workers=half)
    else:
        raise ValueError(f"STAGE1_READ_MODE must be 'middle_cache', 'stream' or 'frame', not {cfg.STAGE1_READ_MODE!r}")
    return train, val, test


def main():
    s = cfg.STAGE1
    resuming = cfg.STAGE1_RESUME_RUN is not None
    if not resuming and cfg.SKIP_FINISHED and run_utils.registered_run("stage1") is not None:
        print(f"Stage 1 is already registered ({run_utils.registered_run('stage1')}); set SKIP_FINISHED = False to train again.")
        return
    device = models.device()
    data_loading.set_seed(cfg.STAGE1_SEED)
    run_dir = cfg.RUNS_DIR / cfg.STAGE1_RESUME_RUN if resuming else run_utils.new_run_dir("stage1_appearance", __file__)
    print(f"[Stage 1] device {device}; run folder {run_dir}; reading mode {cfg.STAGE1_READ_MODE}")

    # ---- data: every sequence is used (also those with fewer than 3 frames), as in the A549 Stage 1 run ----
    train_rec = data_loading.load_index(cfg.NCM_INDEX, "train", min_T=1)
    val_rec = data_loading.load_index(cfg.NCM_INDEX, "val", min_T=1)
    test_rec = data_loading.load_index(cfg.NCM_INDEX, "test", min_T=1)
    weight = data_loading.class_weights(train_rec, s["weight_power"]).to(device)
    train_loader, val_loader, test_loader = make_loaders(train_rec, val_rec, test_rec)
    print(f"sequences: train {len(train_rec)}, val {len(val_rec)}, test {len(test_rec)}; frames: train {len(train_loader.dataset)}, "
          f"val {len(val_loader.dataset)}, test {len(test_loader.dataset)}")

    # ---- model, optimiser, loss ----
    net = models.build_stage1().to(device)
    optimizer = Adam(net.parameters(), lr=s["base_lr"] * s["batch_size"] / 256, weight_decay=s["weight_decay"])
    schedule = run_utils.WarmupCosineSchedule(optimizer, s["warmup_epochs"], s["epochs"], s["min_lr"])
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=s["label_smoothing"])
    scaler = GradScaler(enabled=torch.cuda.is_available())
    writer = run_utils.make_writer(run_dir)

    best_top1, start_epoch = -1.0, 0
    if resuming:
        state = torch.load(run_dir / "resume_state.pt", map_location=device, weights_only=False)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        best_top1, start_epoch = state["best_top1"], state["epoch"]
        print(f"continuing after epoch {start_epoch}; best validation top-1 so far {best_top1:.2f}%")

    for epoch in range(start_epoch, s["epochs"]):
        started = time.time()
        schedule.step(epoch)
        if hasattr(train_loader.dataset, "epoch"):
            train_loader.dataset.epoch = epoch                       # new sequence order and buffer mixing in every epoch
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
        writer.add_scalar("Val/MacroF1", macro_f1, epoch)
        print(f"Epoch {epoch + 1}: val top-1 {top1:.2f}%  macro-F1 {macro_f1:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}  "
              f"{(time.time() - started) / 60:.1f} min")
        torch.save(net.state_dict(), run_dir / "last_model.pth")
        if top1 > best_top1:
            best_top1 = top1
            torch.save(net.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth (top-1 {best_top1:.2f}%)")
        torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch + 1,
                        best_top1=best_top1), run_dir / "resume_state.pt")
    writer.close()

    # ---- predictions of the best checkpoint for validation and test frames ----
    net.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    rows = []
    for name, loader in (("val", val_loader), ("test", test_loader)):
        top1, macro_f1, labels, preds, pmax, top5 = evaluate_frames(net, loader, device)
        print(f"[{name}] frames: top-1 {top1:.2f}%  macro-F1 {macro_f1:.4f}")
        rows += [dict(split=name, label=int(l), pred=int(p), pmax=float(m), top5_correct=bool(t)) for l, p, m, t in zip(labels, preds, pmax, top5)]
    pd.DataFrame(rows).to_csv(run_dir / "stage1_export.csv", index=False)
    run_utils.register_run("stage1", run_dir)
    print(f"Done. Stage 1 saved to {run_dir}")


if __name__ == "__main__":
    main()
