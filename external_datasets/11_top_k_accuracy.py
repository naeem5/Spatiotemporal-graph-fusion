# top-1, top-3 and top-5 accuracy of the appearance expert, the graph expert and the gated fusion, from the stored test outputs. CPU only, no images, no GPU.
import os

import numpy as np
import pandas as pd

import config as cfg
import evaluation

KS = (1, 3, 5)
VARIANTS = ("appearance", "graph", "fused")


def hits_at_k(scores, labels):
    # (N, len(KS)) bool: is the true class among the k highest scores.
    rank = (scores > scores[np.arange(len(labels)), labels][:, None]).sum(axis=1)       # number of classes with a higher score
    return np.stack([rank < k for k in KS], axis=1)


def main():
    protocol = cfg.PROTOCOL if "PUBLIC_POOL_PROTOCOL" not in os.environ else os.environ["PUBLIC_POOL_PROTOCOL"]
    folds = [int(f) for f in os.environ.get("PUBLIC_POOL_FOLDS", "0,1,2,3,4" if cfg.IS_OFFICIAL_FOLDS else "0,1,2" if protocol == "plate" else "0,1,2,3").split(",")]
    parts = [evaluation.load_fold_scores(protocol, f) for f in folds]
    scales = [s for s, _ in parts]
    data = {k: np.concatenate([d[k] for _, d in parts]) for k in ("labels", "group", *VARIANTS)}
    labels, n_classes = data["labels"], cfg.num_classes()
    hits = {v: hits_at_k(data[v], labels) for v in VARIANTS}
    ids, inverse = np.unique(evaluation.cluster_ids(data["group"]), return_inverse=True)

    lines = [f"# Top-k accuracy: {cfg.DATASET}, {protocol}{cfg.FRAMES_TAG}, folds {folds}", "",
             "Chosen s per fold (validation only): " + ", ".join(map(str, scales)), "",
             f"Test units pooled: {len(labels)}; classes: {n_classes}; clusters for the bootstrap: {len(ids)}.", ""]
    rows = [dict(variant=v, **{f"top-{k}": 100.0 * hits[v][:, i].mean() for i, k in enumerate(KS)}) for v in VARIANTS]
    rows.append(dict(variant="chance", **{f"top-{k}": 100.0 * min(k, n_classes) / n_classes for k in KS}))
    lines += ["## Pooled accuracy (%)", "", "```", pd.DataFrame(rows).round(2).to_string(index=False), "```", ""]

    per_cluster = {v: np.zeros((len(ids), len(KS))) for v in VARIANTS}
    for v in VARIANTS:
        np.add.at(per_cluster[v], inverse, hits[v].astype(float))
    size = np.bincount(inverse, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(cfg.BOOT_SEED)
    comparisons = (("fused", "appearance"), ("fused", "graph"))
    draws = {c: np.empty((cfg.N_BOOT, len(KS))) for c in comparisons}
    for b in range(cfg.N_BOOT):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        for new, base in comparisons:
            draws[(new, base)][b] = 100.0 * (w @ (per_cluster[new] - per_cluster[base])) / (w @ size)
    out = []
    for new, base in comparisons:
        for i, k in enumerate(KS):
            est = 100.0 * (hits[new][:, i].mean() - hits[base][:, i].mean())
            out.append(dict(comparison=f"{new} minus {base}", k=k, gain=est, ci_low=np.percentile(draws[(new, base)][:, i], 2.5),
                            ci_high=np.percentile(draws[(new, base)][:, i], 97.5)))
    lines += ["## Gain in percentage points (95% bootstrap interval over clusters)", "", "```", pd.DataFrame(out).round(2).to_string(index=False), "```"]
    text = "\n".join(lines)
    path = cfg.REPORT_BASE / f"{cfg.DATASET}_{protocol}_topk{cfg.ANCHOR_TAG}{cfg.FRAMES_TAG}{cfg.ARCH_TAG}{cfg.PURITY_TAG}.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
