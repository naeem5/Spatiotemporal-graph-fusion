# Stage 3, the class-wise gate on the two frozen experts, trained on NCM460 (one run per seed).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
import data_loading
import evaluation
import models
import run_utils


def to_device(feats, device):
    # Stored expert outputs of a split as tensors on the device (a few MB, the whole split fits easily).
    tensors = {k: torch.from_numpy(np.asarray(feats[k])).to(device) for k in evaluation.FEATURE_KEYS}
    tensors["labels"] = torch.from_numpy(np.asarray(feats["labels"])).long().to(device)
    return tensors


@torch.no_grad()
def validate(model, val, batch_size=1024):
    # Predictions of the fused model, Stage 1 alone and Stage 2 alone on the validation outputs, and the mean gate.
    model.eval()
    fused, app, graph, gates = [], [], [], []
    for i in range(0, len(val["labels"]), batch_size):
        batch = {k: val[k][i:i + batch_size] for k in evaluation.FEATURE_KEYS}
        logits, logits_app, delta, gate, _, _ = model.fuse(batch)
        fused.append(logits.argmax(1)), app.append(logits_app.argmax(1)), graph.append(delta.argmax(1)), gates.append(gate)
    cat = lambda v: torch.cat(v).cpu().numpy()
    return cat(fused), cat(app), cat(graph), float(torch.cat(gates).mean())


def train_one_seed(seed, device, train, val, weight):
    s = cfg.STAGE3
    app_run, graph_run = run_utils.registered_run("stage1"), run_utils.registered_run("stage2")
    data_loading.set_seed(seed)
    run_dir = run_utils.new_run_dir("stage3_gate", __file__, seed=seed)
    print(f"\n[Stage 3] seed {seed}; run folder {run_dir}\n  Stage 1: {app_run.name}\n  Stage 2: {graph_run.name}")

    model = models.build_stage3(app_run / "best_model.pth", graph_run / "best_model.pth").to(device)
    trainable = [(n, p) for n, p in model.named_parameters() if n.startswith("gate_head.") or n == "prior_beta_raw"]
    unexpected = [n for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("gate_head.") or n == "prior_beta_raw")]
    if unexpected:
        raise RuntimeError("unexpected trainable parameters: " + ", ".join(unexpected))
    print(f"  trainable parameters (gate only): {sum(p.numel() for _, p in trainable):,}")
    optimizer = run_utils.build_optimizer(trainable, s["lr"], s["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=s["lr_patience"], min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=s["label_smoothing"])
    writer = run_utils.make_writer(run_dir)
    order = torch.Generator().manual_seed(seed)
    n, bs = len(train["labels"]), s["batch_size"]
    val_labels = val["labels"].cpu().numpy()

    best_score, no_improve = -1.0, 0
    for epoch in range(s["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=order).to(device)
        for i in range(0, n - bs + 1, bs):                       # incomplete last batch dropped, as in the A549 runs
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            logits = model.fuse({k: train[k][idx] for k in evaluation.FEATURE_KEYS})[0]
            criterion(logits, train["labels"][idx]).backward()
            optimizer.step()

        p_fused, p_app, p_graph, mean_gate = validate(model, val)
        top1, macro_f1 = run_utils.top1_and_macro_f1(val_labels, p_fused)
        top1_app, f1_app = run_utils.top1_and_macro_f1(val_labels, p_app)
        top1_graph, f1_graph = run_utils.top1_and_macro_f1(val_labels, p_graph)
        writer.add_scalar("Val/Top1", top1, epoch)
        writer.add_scalar("Val/MacroF1", macro_f1, epoch)
        print(f"Epoch {epoch + 1}: Stage 1 {top1_app:.2f}%/{f1_app:.4f}  Stage 2 {top1_graph:.2f}%/{f1_graph:.4f}  fused {top1:.2f}%/{macro_f1:.4f}  "
              f"mean gate {mean_gate:.3f}  beta {float(F.softplus(model.prior_beta_raw.detach())):.3f}")
        torch.save(model.state_dict(), run_dir / "last_model.pth")
        score = macro_f1 if s["selection_metric"] == "macro_f1" else top1
        scheduler.step(score)
        if score > best_score:
            best_score, no_improve = score, 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            print(f"  saved best_model.pth ({s['selection_metric']} {best_score:.4f})")
        else:
            no_improve += 1
            if no_improve >= s["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    writer.close()
    run_utils.register_run("stage3", run_dir, seed=seed)
    print(f"Done. Stage 3 (seed {seed}) saved to {run_dir}")


def main():
    device = models.device()
    app_run, graph_run = run_utils.registered_run("stage1"), run_utils.registered_run("stage2")
    if app_run is None or graph_run is None:
        raise RuntimeError("Stage 1 and Stage 2 must be trained first (02_train_stage1_appearance.py and 03_train_stage2_graph.py).")
    records = {split: data_loading.load_index(cfg.NCM_INDEX, split, min_T=cfg.MIN_T) for split in ("train", "val", "test")}
    for split, recs in records.items():
        data_loading.check_cache(recs, "Stage 3")
    print(f"sequences with >= {cfg.MIN_T} frames: " + ", ".join(f"{k} {len(v)}" for k, v in records.items()))

    # one pass over the images with the frozen experts (skipped when the stored outputs are up to date)
    extractor = models.build_stage3(app_run / "best_model.pth", graph_run / "best_model.pth").to(device)
    feats = {split: evaluation.cached_features(split, records[split], extractor, device) for split in ("train", "val", "test")}
    del extractor
    train, val = to_device(feats["train"], device), to_device(feats["val"], device)
    weight = data_loading.class_weights(records["train"], cfg.STAGE3["weight_power"]).to(device)

    for seed in cfg.SEEDS:
        if cfg.SKIP_FINISHED and run_utils.registered_run("stage3", seed) is not None:
            print(f"Seed {seed} is already registered ({run_utils.registered_run('stage3', seed).name}); skipped.")
            continue
        train_one_seed(seed, device, train, val, weight)


if __name__ == "__main__":
    main()
