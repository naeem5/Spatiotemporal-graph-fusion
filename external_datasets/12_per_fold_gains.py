# gain of the gated fusion over the appearance expert in every fold, from the stored test outputs. CPU only, no images, no GPU.
import pandas as pd

import config as cfg
import evaluation
import run_utils

# (protocol, folds, [(graph expert sees, frames_tag)])
SETTINGS = {"lctp_a549": [("plate", (0, 1, 2), [("sequence", ""), ("single frame", "_1frame")]), ("compound", (0, 1, 2, 3), [("sequence", "")])],
            "pastis": [("official5", (0, 1, 2, 3, 4), [("sequence", ""), ("single frame", "_1frame")])],
            "hyperkvasir": [("official5", (0, 1, 2, 3, 4), [("single frame", "")])]}
OUT = cfg.REPORT_BASE / "external_per_fold_gains.csv"


def main():
    rows = []
    for protocol, folds, variants in SETTINGS[cfg.DATASET]:
        for sees, tag in variants:
            for fold in folds:
                try:
                    _, d = evaluation.load_fold_outputs(protocol, fold, frames_tag=tag)
                except (FileNotFoundError, OSError):
                    print(f"missing: {cfg.DATASET} {protocol} fold {fold} {tag or 'all frames'}")
                    continue
                top_a, f1_a = run_utils.top1_and_macro_f1(d["labels"], d["appearance"])
                top_f, f1_f = run_utils.top1_and_macro_f1(d["labels"], d["fused"])
                rows.append(dict(dataset=cfg.DATASET, protocol=protocol, graph_sees=sees, fold=fold, n=len(d["labels"]),
                                 top1_points=top_f - top_a, macro_f1_x100=100.0 * (f1_f - f1_a)))
    new = pd.DataFrame(rows)
    print(new.round(3).to_string(index=False))
    if OUT.exists():
        old = pd.read_csv(OUT)
        keep = old[old.dataset != cfg.DATASET]
        new = pd.concat([keep, new], ignore_index=True)
    new.to_csv(OUT, index=False)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

