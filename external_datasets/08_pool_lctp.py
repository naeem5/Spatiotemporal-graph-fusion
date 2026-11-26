# pooled result of the LCTP plate protocol over its folds, and a compound-level bootstrap. CPU only, reads the stored test outputs (no images, no GPU).
import os

import numpy as np
import pandas as pd

import config as cfg
import evaluation
import run_utils


def main():
    protocol = os.environ.get("PUBLIC_POOL_PROTOCOL", "plate")
    folds = [int(f) for f in os.environ.get("PUBLIC_POOL_FOLDS", "0,1,2").split(",")]
    idx = pd.read_csv(cfg.INDEX)
    well_to_compound = dict(idx[["well", "compound"]].drop_duplicates().itertuples(index=False))
    control = cfg.load_classes()["names"][cfg.load_classes()["control_label"]]
    control_wells = set(idx.loc[idx["moa"] == control, "well"])                       # DMSO wells: independent replicates of the control, not a "compound"
    parts, lines, scales = {}, [f"# Pooled {cfg.DATASET} {protocol} protocol, folds {folds}", ""], {}
    for f in folds:
        scales[f], parts[f] = evaluation.load_fold_outputs(protocol, f)
        parts[f]["compound"] = np.array([well_to_compound[w] for w in parts[f]["group"]])
        parts[f]["compound_dmso_by_well"] = np.array([w if w in control_wells else well_to_compound[w] for w in parts[f]["group"]])
    pooled = {k: np.concatenate([parts[f][k] for f in folds]) for k in ("labels", "group", "appearance", "graph", "fused", "compound", "compound_dmso_by_well")}
    parts["pooled"] = pooled
    lines += ["Chosen s per fold (validation only): " + ", ".join(f"fold {f}: {scales[f]}" for f in folds), ""]

    rows = []
    for name, d in parts.items():
        for variant in ("appearance", "graph", "fused"):
            top1, f1 = run_utils.top1_and_macro_f1(d["labels"], d[variant])
            rows.append(dict(fold=name, variant=variant, n=len(d["labels"]), top1=top1, macro_f1=f1))
    tab = pd.DataFrame(rows)
    lines += ["## Test performance", "", "```", tab.round(3).to_string(index=False), "```", ""]

    gains = []
    for name, d in parts.items():
        for other in ("appearance", "graph"):
            for unit, ids in (("wells", d["group"]), ("compounds", d["compound"]), ("compounds, DMSO by well", d["compound_dmso_by_well"])):
                g = evaluation.group_bootstrap_gain(d["labels"], d[other], d["fused"], ids)
                fixed, broken, p = evaluation.exact_mcnemar(d["labels"], d[other], d["fused"])
                gains.append(dict(fold=name, comparison=f"fused vs {other}", ci_units=unit, groups=g["n_groups"], top1_gain=g["top1_gain"], top1_low=g["top1_ci"][0],
                                  top1_high=g["top1_ci"][1], f1_gain=g["f1_gain"], f1_low=g["f1_ci"][0], f1_high=g["f1_ci"][1], fixed=fixed, broken=broken, mcnemar_p=p))
    lines += ["## Gain of the fusion", "", "```", pd.DataFrame(gains).round(4).to_string(index=False), "```", "",
              "ci_units: what was resampled for the 95% percentile interval. mcnemar_p assumes independent tiles and is therefore optimistic; quote the compound-level intervals (both versions).", ""]
    text = "\n".join(lines)
    out = cfg.REPORT_BASE / f"{cfg.DATASET}_{protocol}_pooled{cfg.FRAMES_TAG}{cfg.ARCH_TAG}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
