# does the benefit of the gate depend on the time axis? CPU only, reads the stored test outputs (no images, no GPU).
import os

import numpy as np
import pandas as pd

import config as cfg
import evaluation
import run_utils
from pdgraph.metrics import macro_f1_from_confusion

SETTINGS = {"all frames": "", "single frame": "_1frame"}


def cluster_confusions(labels, preds, clusters):
    # (clusters, K, K) counts of true class x predicted class per cluster, and the cluster ids.
    ids, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    k = cfg.num_classes()
    cm = np.zeros((len(ids), k, k))
    np.add.at(cm, (inverse, np.asarray(labels), np.asarray(preds)), 1)
    return cm, ids


def metrics(cm):
    return 100.0 * np.trace(cm) / cm.sum(), macro_f1_from_confusion(cm)


def paired_cluster_bootstrap(cms, n_boot, seed):
    # cms: {(setting, variant): (clusters, K, K)}. Percentile intervals of the quantities of the report over whole-cluster resamples shared by all settings.
    n = next(iter(cms.values())).shape[0]
    rng = np.random.default_rng(seed)

    def quantities(weights):
        m = {key: metrics(np.tensordot(weights, cm, 1)) for key, cm in cms.items()}
        q = {}
        for i, name in enumerate(("top-1", "macro-F1")):
            gain = {s: m[(s, "fused")][i] - m[(s, "appearance")][i] for s in SETTINGS}
            q[f"{name}: gain of the fusion over the appearance expert, all frames"] = gain["all frames"]
            q[f"{name}: gain of the fusion over the appearance expert, single frame"] = gain["single frame"]
            q[f"{name}: DIFFERENCE of the two gains (all frames minus single frame)"] = gain["all frames"] - gain["single frame"]
            q[f"{name}: graph expert alone, all frames minus single frame"] = m[("all frames", "graph")][i] - m[("single frame", "graph")][i]
            q[f"{name}: fusion, all frames minus single frame"] = m[("all frames", "fused")][i] - m[("single frame", "fused")][i]
        return q

    point = quantities(np.ones(n))
    draws = {k: np.empty(n_boot) for k in point}
    for b in range(n_boot):
        for k, v in quantities(np.bincount(rng.integers(0, n, n), minlength=n).astype(float)).items():
            draws[k][b] = v
    return pd.DataFrame([dict(quantity=k, estimate=point[k], ci_low=np.percentile(draws[k], 2.5), ci_high=np.percentile(draws[k], 97.5),
                              share_of_resamples_le_0=float((draws[k] <= 0).mean())) for k in point])


def main():
    protocol = cfg.PROTOCOL if "PUBLIC_POOL_PROTOCOL" not in os.environ else os.environ["PUBLIC_POOL_PROTOCOL"]
    folds = [int(f) for f in os.environ.get("PUBLIC_POOL_FOLDS", "0,1,2,3,4" if cfg.IS_OFFICIAL_FOLDS else "0,1,2" if protocol == "plate" else "0,1,2,3").split(",")]
    scales = {s: {} for s in SETTINGS}
    data = {s: [] for s in SETTINGS}
    for f in folds:
        loaded = {s: evaluation.load_fold_outputs(protocol, f, frames_tag=tag) for s, tag in SETTINGS.items()}
        (_, a), (_, b) = loaded["all frames"], loaded["single frame"]
        if not (np.array_equal(a["labels"], b["labels"]) and np.array_equal(a["group"], b["group"])):
            raise RuntimeError(f"fold {f}: the two settings were not evaluated on the same test units in the same order")
        for s in SETTINGS:
            scales[s][f] = loaded[s][0]
            data[s].append(loaded[s][1])
    pooled = {s: {k: np.concatenate([d[k] for d in data[s]]) for k in ("labels", "group", "appearance", "graph", "fused")} for s in SETTINGS}
    labels, group = pooled["all frames"]["labels"], pooled["all frames"]["group"]
    clusters = evaluation.cluster_ids(group)

    lines = [f"# {cfg.DATASET}, {protocol}{cfg.ANCHOR_TAG}, folds {folds}: graph expert with all frames versus the anchor frame only", "",
             "Chosen s per fold (validation only): " + "; ".join(f"{s}: " + ", ".join(f"fold {f} {scales[s][f]}" for f in folds) for s in SETTINGS), "",
             f"Test units pooled: {len(labels)} (each unit once); clusters for the bootstrap: {len(set(clusters))}.", ""]
    rows = []
    for s in SETTINGS:
        for v in ("appearance", "graph", "fused"):
            top1, f1 = run_utils.top1_and_macro_f1(labels, pooled[s][v])
            rows.append(dict(graph_expert_sees=s, variant=v, top1=top1, macro_f1=f1))
    lines += ["## Test performance (pooled)", "", "```", pd.DataFrame(rows).round(3).to_string(index=False), "```", ""]
    differ = int((pooled["all frames"]["appearance"] != pooled["single frame"]["appearance"]).sum())
    lines += [f"Units on which the appearance expert of the two settings predicts differently: {differ} of {len(labels)} "
              "(same frame, same seed: expected 0 or very few).", ""]

    cms = {(s, v): cluster_confusions(labels, pooled[s][v], clusters)[0] for s in SETTINGS for v in ("appearance", "graph", "fused")}
    table = paired_cluster_bootstrap(cms, cfg.N_BOOT, cfg.BOOT_SEED)
    lines += ["## Paired comparison, 95% bootstrap over clusters", "", "```", table.round(4).to_string(index=False), "```", "",
              "The row that answers the question is 'DIFFERENCE of the two gains'. If its interval excludes 0 and the gain with a single frame is about 0, the benefit of the gate "
              "depends on the time axis (same data, same units, only the frames of the graph expert changed). If the single-frame gain is as large as the all-frames gain, it does not.", ""]
    text = "\n".join(lines)
    out = cfg.REPORT_BASE / f"{cfg.DATASET}_{protocol}{cfg.ANCHOR_TAG}{cfg.PURITY_TAG}_frames_comparison.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
