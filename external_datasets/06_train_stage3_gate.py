# Stage 3, the class-wise gate on the two frozen experts, with the fusion strength s chosen on the VALIDATION set only.
import json

import numpy as np
import torch
import torch.nn as nn

import config as cfg
import data_loading
import evaluation
import models
import run_utils


def to_device(feats, device):
    tensors = {k: torch.from_numpy(np.asarray(feats[k])).to(device) for k in evaluation.FEATURE_KEYS}
    tensors["labels"] = torch.from_numpy(np.asarray(feats["labels"])).long().to(device)
    return tensors


@torch.no_grad()
def validate(model, val, batch_size=1024):
    model.eval()
    fused, gates = [], []
    for i in range(0, len(val["labels"]), batch_size):
        batch = {k: val[k][i:i + batch_size] for k in evaluation.FEATURE_KEYS}
        logits, _, _, gate, _, _ = model.fuse(batch)
        fused.append(logits.argmax(1)), gates.append(gate)
    return torch.cat(fused).cpu().numpy(), float(torch.cat(gates).mean())


def fit_gate(scale, seed, train, val, weight, device):
    # Train the gate for one scale and seed; returns the best gate state and its validation scores.
    s = cfg.STAGE3
    data_loading.set_seed(seed)
    model = models.build_stage3(delta_scale=scale).to(device)         # the experts are not used here: the gate works on stored outputs
    trainable = [(n, p) for n, p in model.named_parameters() if n.startswith("gate_head.") or n == "prior_beta_raw"]
    optimizer = run_utils.build_optimizer(trainable, s["lr"], s["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=s["lr_patience"], min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=s["label_smoothing"])
    order = torch.Generator().manual_seed(seed)
    n, bs = len(train["labels"]), min(s["batch_size"], len(train["labels"]))
    val_labels = val["labels"].cpu().numpy()
    best, wait = dict(f1=-1.0), 0
    for epoch in range(s["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=order).to(device)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            criterion(model.fuse({k: train[k][idx] for k in evaluation.FEATURE_KEYS})[0], train["labels"][idx]).backward()
            optimizer.step()
        fused, mean_gate = validate(model, val)
        top1, f1 = run_utils.top1_and_macro_f1(val_labels, fused)
        scheduler.step(f1)
        if f1 > best["f1"]:
            best = dict(f1=f1, top1=top1, gate=mean_gate, epoch=epoch + 1,
                        state={k: v.detach().clone().cpu() for k, v in model.state_dict().items() if k.startswith("gate_head.") or k == "prior_beta_raw"})
            wait = 0
        else:
            wait += 1
            if wait >= s["patience"]:
                break
    return best


def main():
    if cfg.SKIP_FINISHED and run_utils.registered_run("stage3") is not None:
        print(f"Stage 3 is already registered ({run_utils.registered_run('stage3')}); set SKIP_FINISHED = False to train again.")
        return
    device = models.device()
    app_run, graph_run = run_utils.registered_run("stage1"), run_utils.registered_run("stage2")
    if app_run is None or graph_run is None:
        raise RuntimeError("train Stage 1 and Stage 2 first (04_train_stage1_appearance.py, 05_train_stage2_graph.py)")
    records = {sp: data_loading.load_index(sp, min_T=cfg.MIN_T) for sp in ("train", "val", "test")}
    print(f"[Stage 3] {cfg.RUN_TAG}; sequences with >= {cfg.MIN_T} frames: " + ", ".join(f"{k} {len(v)}" for k, v in records.items()))

    extractor = models.build_stage3(app_run / "best_model.pth", graph_run / "best_model.pth").to(device)
    feats = {sp: evaluation.cached_features(sp, records[sp], extractor, device) for sp in ("train", "val", "test")}
    del extractor
    train, val = to_device(feats["train"], device), to_device(feats["val"], device)
    weight = data_loading.class_weights(records["train"], cfg.STAGE3["weight_power"]).to(device)

    run_dir = run_utils.new_run_dir("stage3_gate", __file__)
    rows, states = [], {}
    for scale in cfg.GATE_SCALE_GRID:
        results = [fit_gate(scale, seed, train, val, weight, device) for seed in cfg.SEEDS]
        states[scale] = [r["state"] for r in results]
        rows.append(dict(scale=scale, val_top1=round(float(np.mean([r["top1"] for r in results])), 2),
                         val_macro_f1=round(float(np.mean([r["f1"] for r in results])), 4), mean_gate=round(float(np.mean([r["gate"] for r in results])), 3)))
        print(f"scale {scale}: val top-1 {rows[-1]['val_top1']}, macro-F1 {rows[-1]['val_macro_f1']}, mean gate {rows[-1]['mean_gate']}")
    best = max(rows, key=lambda r: r["val_macro_f1"])["scale"]
    torch.save(dict(states=states, seeds=cfg.SEEDS), run_dir / "gates.pt")
    (run_dir / "selection.json").write_text(json.dumps(dict(chosen_scale=best, reference_scale=cfg.REFERENCE_SCALE, table=rows), indent=1), encoding="utf-8")
    run_utils.register_run("stage3", run_dir)
    print(f"Chosen scale (best mean validation macro-F1): {best}. Saved to {run_dir}")


if __name__ == "__main__":
    main()
