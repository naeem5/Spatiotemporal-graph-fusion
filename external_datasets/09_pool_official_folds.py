# pooled result of a data set run with the official folds (HyperKvasir, PASTIS): every image or patch is a test unit exactly once. CPU only, reads the stored test outputs.
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config as cfg
import evaluation
import run_utils


def main():
    folds = [int(f) for f in os.environ.get("PUBLIC_POOL_FOLDS", "0,1,2,3,4").split(",")]
    names = cfg.class_names()
    parts, scales = {}, {}
    for f in folds:
        scales[f], parts[f] = evaluation.load_fold_outputs(cfg.PROTOCOL, f)
    pooled = {k: np.concatenate([parts[f][k] for f in folds]) for k in ("labels", "group", "appearance", "graph", "fused")}
    parts["pooled"] = pooled
    tag = ("_noref" if cfg.NO_REFERENCE else "") + cfg.ANCHOR_TAG + cfg.FRAMES_TAG + cfg.ARCH_TAG + cfg.PURITY_TAG
    lines = [f"# Pooled {cfg.DATASET} {cfg.PROTOCOL}{tag}, reference = {cfg.REFERENCE}, folds {folds}", "",
             "Chosen s per fold (validation fold only): " + ", ".join(f"fold {f}: {scales[f]}" for f in folds), "",
             f"Test units: {len(pooled['labels'])}; duplicates among the pooled units: {len(pooled['group']) - len(set(pooled['group']))}", ""]

    rows = []
    for name, d in parts.items():
        for variant in ("appearance", "graph", "fused"):
            top1, f1 = run_utils.top1_and_macro_f1(d["labels"], d[variant])
            rows.append(dict(fold=name, variant=variant, n=len(d["labels"]), top1=top1, macro_f1=f1))
    tab = pd.DataFrame(rows)
    lines += ["## Test performance", "", "```", tab.round(3).to_string(index=False), "```", "",
              "Mean of the per-fold macro-F1 (a fold with a class of one test unit is noisy; the pooled value is the one to quote): " +
              ", ".join(f"{v} {tab[(tab.variant == v) & (tab.fold != 'pooled')]['macro_f1'].mean():.3f}" for v in ("appearance", "graph", "fused")), ""]

    gains = []
    for name, d in parts.items():
        for other in ("appearance", "graph"):
            g = evaluation.group_bootstrap_gain(d["labels"], d[other], d["fused"], d["group"])
            fixed, broken, p = evaluation.exact_mcnemar(d["labels"], d[other], d["fused"])
            gains.append(dict(fold=name, comparison=f"fused vs {other}", units=g["n_groups"], top1_gain=g["top1_gain"], top1_low=g["top1_ci"][0], top1_high=g["top1_ci"][1],
                              f1_gain=g["f1_gain"], f1_low=g["f1_ci"][0], f1_high=g["f1_ci"][1], fixed=fixed, broken=broken, mcnemar_p=p))
    lines += ["## Gain of the fusion (95% bootstrap interval over units)", "", "```", pd.DataFrame(gains).round(4).to_string(index=False), "```", ""]

    k = len(names)
    per_class = pd.DataFrame({"class": names, "n": np.bincount(pooled["labels"], minlength=k)})
    for variant in ("appearance", "graph", "fused"):
        per_class[f"F1 {variant}"] = f1_score(pooled["labels"], pooled[variant], labels=list(range(k)), average=None, zero_division=0)
    lines += ["## F1 per class (pooled)", "", "```", per_class.round(3).to_string(index=False), "```"]
    text = "\n".join(lines)
    out = cfg.REPORT_BASE / f"{cfg.DATASET}_{cfg.PROTOCOL}_pooled{tag}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
