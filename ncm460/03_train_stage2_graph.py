# Stage 2, the spatiotemporal patch-graph expert, trained on NCM460.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm

import config as cfg
import data_loading
import models
import run_utils
from pdgraph.graph_expert import GRAPH_STAT_KEYS


@torch.no_grad()
def calibrate_control_reference(model, control_loader, device):
    # Mean of each descriptor over the control sequences of the training set, measured in eval mode, stored in the model as the reference.
    model.eval()
    sums = {k: torch.zeros((), dtype=torch.float64, device=device) for k in GRAPH_STAT_KEYS}
    count = 0
    for x, mask, _, _ in tqdm(control_loader, desc="[control reference]", leave=False):
        _, stats = model.encoder(x.to(device), mask.to(device))
        for k in GRAPH_STAT_KEYS:
            sums[k] += stats[k].double().sum()
        count += x.size(0)
    if count == 0:
        raise RuntimeError("no control sequences in the training split: the control reference cannot be measured")
    means = {k: (sums[k] / count).float() for k in GRAPH_STAT_KEYS}
    model.control_ref.set_reference(means)
    return {k: round(float(v), 4) for k, v in means.items()}


@torch.no_grad()
def evaluate_sequences(model, loader, device, keep_stats=False):
    # Top-1, top-5, macro-F1 of a loader of sequences; optionally the descriptors of every sequence for the export.
    model.eval()
    labels, preds, top5, rows = [], [], [], []
    for x, mask, lengths, y in tqdm(loader, desc="[evaluate]", leave=False):
        logits, _, stats, dev_stats = model(x.to(device), mask.to(device))
        labels.append(y.numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        top5.append((logits.topk(min(5, cfg.NUM_CLASSES), dim=1).indices.cpu() == y.unsqueeze(1)).any(1).numpy())
        if keep_stats:
            for i in range(x.size(0)):
                row = {"label": int(y[i]), "pred": int(preds[-1][i]), "T": int(lengths[i])}
                for k in GRAPH_STAT_KEYS:
                    row[k] = float(stats[k][i])
                    row[f"{k}_dev"] = float(dev_stats[k][i])
                rows.append(row)
    labels, preds = np.concatenate(labels), np.concatenate(preds)
    top1, macro_f1 = run_utils.top1_and_macro_f1(labels, preds)
    return top1, 100.0 * float(np.concatenate(top5).mean()), macro_f1, preds, rows


def main():
    s = cfg.STAGE2
    if cfg.SKIP_FINISHED and run_utils.registered_run("stage2") is not None:
        print(f"Stage 2 is already registered ({run_utils.registered_run('stage2')}); set SKIP_FINISHED = False to train again.")
        return
    device = models.device()
    data_loading.set_seed(cfg.STAGE2_SEED)
    run_dir = run_utils.new_run_dir("stage2_graph", __file__)
    print(f"[Stage 2] device {device}; run folder {run_dir}")

    # ---- data ----
    train_rec = data_loading.load_index(cfg.NCM_INDEX, "train", min_T=cfg.MIN_T)
    val_rec = data_loading.load_index(cfg.NCM_INDEX, "val", min_T=cfg.MIN_T)
    test_rec = data_loading.load_index(cfg.NCM_INDEX, "test", min_T=cfg.MIN_T)
    for records in (train_rec, val_rec, test_rec):
        data_loading.check_cache(records, "Stage 2")
    control_rec = [r for r in train_rec if r["label"] == cfg.CONTROL_LABEL]
    weight = data_loading.class_weights(train_rec, s["weight_power"]).to(device)
    half = max(cfg.NUM_WORKERS // 2, 0)
    train_loader = data_loading.make_loader(data_loading.GraphSequenceDataset(train_rec), s["batch_size"], True, True, data_loading.collate_graph, cfg.STAGE2_SEED)
    val_loader = data_loading.make_loader(data_loading.GraphSequenceDataset(val_rec), s["batch_size"], False, False, data_loading.collate_graph, workers=half)
    test_loader = data_loading.make_loader(data_loading.GraphSequenceDataset(test_rec), s["batch_size"], False, False, data_loading.collate_graph, workers=half)
    control_loader = data_loading.make_loader(data_loading.GraphSequenceDataset(control_rec), s["batch_size"], False, False, data_loading.collate_graph, workers=half)
    print(f"sequences with >= {cfg.MIN_T} frames: train {len(train_rec)} (controls {len(control_rec)}), val {len(val_rec)}, test {len(test_rec)}")

    # ---- model, optimiser, loss ----
    model = models.build_stage2().to(device)
    optimizer = run_utils.build_optimizer(model.named_parameters(), s["base_lr"] * s["batch_size"] / 64, s["weight_decay"])
    schedule = run_utils.WarmupCosineSchedule(optimizer, max(5, s["epochs"] // 10), s["epochs"], s["min_lr"])
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=s["label_smoothing"])
    scaler = GradScaler(enabled=torch.cuda.is_available())
    writer = run_utils.make_writer(run_dir)

    best_score, best_epoch, no_improve = -1.0, 0, 0
    for epoch in range(s["epochs"]):
        schedule.step(epoch)
        model.train()
        for x, mask, _, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{s['epochs']} [train]"):
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                # the batch also updates the running control reference with its control sequences
                logits, _, _, _ = model(x, mask, is_control=(y == cfg.CONTROL_LABEL), update_reference=True)
                loss = criterion(logits, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss in epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=s["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

        reference = calibrate_control_reference(model, control_loader, device)    # reference in eval mode, then validate with it
        top1, top5, macro_f1, preds, _ = evaluate_sequences(model, val_loader, device)
        writer.add_scalar("Val/Top1", top1, epoch)
        writer.add_scalar("Val/MacroF1", macro_f1, epoch)
        print(f"Epoch {epoch + 1}: val top-1 {top1:.2f}%  top-5 {top5:.2f}%  macro-F1 {macro_f1:.4f}  lr {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  predicted counts {np.bincount(preds, minlength=cfg.NUM_CLASSES).tolist()}  control reference {reference}")
        torch.save(model.state_dict(), run_dir / "last_model.pth")
        score = macro_f1 if s["selection_metric"] == "macro_f1" else top1
        if score > best_score:
            best_score, best_epoch, no_improve = score, epoch + 1, 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth ({s['selection_metric']} {best_score:.4f})")
        else:
            no_improve += 1
            if no_improve >= s["patience"]:
                print(f"Early stopping at epoch {epoch + 1}; best epoch {best_epoch} ({s['selection_metric']} {best_score:.4f})")
                break
    writer.close()

    # ---- best checkpoint: result on val and test, and the descriptors of every sequence ----
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    export = []
    for name, loader in (("val", val_loader), ("test", test_loader)):
        top1, top5, macro_f1, _, rows = evaluate_sequences(model, loader, device, keep_stats=True)
        print(f"[{name}] top-1 {top1:.2f}%  top-5 {top5:.2f}%  macro-F1 {macro_f1:.4f}")
        export += [{"split": name, **r} for r in rows]
    pd.DataFrame(export).to_csv(run_dir / "stage2_export.csv", index=False)
    run_utils.register_run("stage2", run_dir)
    print(f"Done. Stage 2 saved to {run_dir}")


if __name__ == "__main__":
    main()
