# test results of the registered Stage 1, 2, 3 runs of a public data set. Reads the stored expert outputs (no images are read again).
import json

import numpy as np
import pandas as pd
import torch

import config as cfg
import data_loading
import evaluation
import models
import run_utils


def as_markdown(df):
    return "```\n" + df.round(3).to_string(index=False) + "\n```"


def load_selection():
    run = run_utils.registered_run("stage3")
    if run is None:
        raise RuntimeError("train Stage 3 first (06_train_stage3_gate.py)")
    return json.loads((run / "selection.json").read_text()), torch.load(run / "gates.pt", map_location="cpu", weights_only=False), run


def outputs_for_scale(scale, gates, feats, device):
    outs = []
    for state in gates["states"][scale]:
        model = models.build_stage3(delta_scale=scale).to(device)
        model.load_state_dict(state, strict=False)
        outs.append(evaluation.fused_outputs(model, feats, device))
    return evaluation.ensemble_outputs(outs)


def gain_rows(out, scale, label):
    preds = evaluation.prediction_variants(out, scale)
    rows = []
    for other in ("appearance", "graph"):
        g = evaluation.group_bootstrap_gain(out["labels"], preds[other], preds["fused"], out["group"])
        fixed, broken, p = evaluation.exact_mcnemar(out["labels"], preds[other], preds["fused"])
        rows.append(dict(comparison=f"{label}: fused vs {other}", groups=g["n_groups"], top1_gain=g["top1_gain"], top1_low=g["top1_ci"][0], top1_high=g["top1_ci"][1],
                         f1_gain=g["f1_gain"], f1_low=g["f1_ci"][0], f1_high=g["f1_ci"][1], fixed=fixed, broken=broken, mcnemar_p=p))
    return pd.DataFrame(rows)


def main():
    device = models.device()
    selection, gates, run = load_selection()
    best, ref = float(selection["chosen_scale"]), float(selection["reference_scale"])
    test_rec = data_loading.load_index("test", min_T=cfg.MIN_T)
    app_run, graph_run = run_utils.registered_run("stage1"), run_utils.registered_run("stage2")
    extractor = models.build_stage3(app_run / "best_model.pth", graph_run / "best_model.pth").to(device)
    feats = evaluation.cached_features("test", test_rec, extractor, device)
    del extractor

    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"# Results: {cfg.RUN_TAG}", "", f"Test sequences: {len(feats['labels'])}; classes: {cfg.num_classes()}; split column: {cfg.SPLIT_COLUMN}",
             f"Stage 1: {app_run.name}; Stage 2: {graph_run.name}; Stage 3: {run.name}", "",
             "## Strength of the graph correction, chosen on the validation set (mean of the seeds)", "", as_markdown(pd.DataFrame(selection["table"])), "",
             f"Chosen s = {best} (best validation macro-F1). Reference s = {ref} (A549 value).", ""]
    arrays, gains = {}, []
    for label, scale in (("chosen s", best), ("reference s", ref)):
        out = outputs_for_scale(scale, gates, feats, device)
        arrays[scale] = out
        lines += [f"## Test set, {label} = {scale}", "", as_markdown(evaluation.summary_table(out, scale)), ""]
        gains.append(gain_rows(out, scale, f"s={scale}"))
    lines += ["## Gain of the fusion", "", as_markdown(pd.concat(gains)), "",
              "top1_gain / f1_gain: fused minus the other variant (percentage points / macro-F1); low/high: 95% whole-group bootstrap interval; "
              "fixed / broken: sequences the fusion gets right / wrong that the other variant gets wrong / right; mcnemar_p: exact two-sided test."]
    (cfg.REPORT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    items = {}
    for scale, out in arrays.items():
        for k, v in out.items():
            items[f"s{scale}_{k}"] = v
        for name, p in evaluation.prediction_variants(out, scale).items():
            items[f"s{scale}_preds_{name}"] = p
    np.savez(cfg.REPORT_DIR / "test_outputs.npz", **items)
    print("\n".join(lines))
    print(f"\nwritten: {cfg.REPORT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
