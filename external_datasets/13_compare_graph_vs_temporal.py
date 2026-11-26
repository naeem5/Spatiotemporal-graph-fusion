# Does the patch graph add to the fusion beyond access to the sequence? Compares the gated fusion built on the patch graph with the fusion built on
# temporal-only control experts (same units, folds, frozen Stage 1 and gate training). CPU only, reads the test outputs stored by 07_evaluate.py.
# Example: PUBLIC_DATASET=lctp_a549 PUBLIC_PROTOCOL=compound PUBLIC_POOL_FOLDS=0,1,2,3 python 13_compare_graph_vs_temporal.py
#          PUBLIC_COMPARE_ARMS=temporal,temporal_nodes selects the control experts (default: temporal)
import os

import numpy as np
import pandas as pd

import config as cfg
import evaluation
import run_utils
from pdgraph.metrics import macro_f1_from_confusion

REFERENCE_ARM = "patch graph"
ARM_TAGS = {"temporal": "_temporal", "temporal_nodes": "_temporal_nodes"}
ARM_NAMES = {"temporal": "temporal (one vector per frame)", "temporal_nodes": "temporal (nodes, no spatial edges)"}


def cluster_confusions(labels, predictions, clusters):
    # (clusters, classes, classes) counts of true class by predicted class
    _, inverse = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    n_classes = cfg.num_classes()
    confusion = np.zeros((inverse.max() + 1, n_classes, n_classes))
    np.add.at(confusion, (inverse, np.asarray(labels), np.asarray(predictions)), 1)
    return confusion


def top1_and_macro_f1_of_confusion(confusion):
    return 100.0 * np.trace(confusion) / confusion.sum(), macro_f1_from_confusion(confusion)


def paired_cluster_bootstrap(confusions, controls, n_boot, seed):
    # confusions: {(arm, variant): per-cluster confusion}, variant in appearance / graph / fused; the resampled clusters are shared by all arms
    n_clusters = next(iter(confusions.values())).shape[0]
    rng = np.random.default_rng(seed)
    arms = [REFERENCE_ARM] + controls

    def quantities(weights):
        metrics = {key: top1_and_macro_f1_of_confusion(np.tensordot(weights, value, 1)) for key, value in confusions.items()}
        result = {}
        for i, metric in enumerate(("top-1", "macro-F1")):
            gain = {arm: metrics[(arm, "fused")][i] - metrics[(arm, "appearance")][i] for arm in arms}
            gain_over_expert = {arm: metrics[(arm, "fused")][i] - metrics[(arm, "graph")][i] for arm in arms}
            for arm in arms:
                result[f"{metric}: gain of the fusion over the appearance expert, {arm_name(arm)}"] = gain[arm]
            for arm in controls:
                result[f"{metric}: DIFFERENCE of the two gains ({REFERENCE_ARM} minus {arm_name(arm)})"] = gain[REFERENCE_ARM] - gain[arm]
                result[f"{metric}: expert alone, {REFERENCE_ARM} minus {arm_name(arm)}"] = metrics[(REFERENCE_ARM, "graph")][i] - metrics[(arm, "graph")][i]
            for arm in arms:
                result[f"{metric}: gain of the fusion over its own expert, {arm_name(arm)}"] = gain_over_expert[arm]
        return result

    estimate = quantities(np.ones(n_clusters))
    draws = {key: np.empty(n_boot) for key in estimate}
    for b in range(n_boot):
        for key, value in quantities(np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters).astype(float)).items():
            draws[key][b] = value
    return pd.DataFrame([dict(quantity=key, estimate=estimate[key], ci_low=np.percentile(draws[key], 2.5), ci_high=np.percentile(draws[key], 97.5),
                              share_of_resamples_le_0=float((draws[key] <= 0).mean())) for key in estimate])


def arm_name(arm):
    return REFERENCE_ARM if arm == REFERENCE_ARM else ARM_NAMES[arm]


def main():
    protocol = os.environ.get("PUBLIC_POOL_PROTOCOL", cfg.PROTOCOL)
    default_folds = "0,1,2,3,4" if cfg.IS_OFFICIAL_FOLDS else "0,1,2" if protocol == "plate" else "0,1,2,3"
    folds = [int(f) for f in os.environ.get("PUBLIC_POOL_FOLDS", default_folds).split(",")]
    controls = [c.strip() for c in os.environ.get("PUBLIC_COMPARE_ARMS", "temporal").split(",") if c.strip()]
    for control in controls:
        if control not in ARM_TAGS:
            raise ValueError(f"PUBLIC_COMPARE_ARMS: {control!r} is not one of {list(ARM_TAGS)}")
    arm_suffix = {REFERENCE_ARM: ""}
    arm_suffix.update({c: ARM_TAGS[c] for c in controls})

    chosen_scale = {arm: {} for arm in arm_suffix}
    fold_outputs = {arm: [] for arm in arm_suffix}
    for fold in folds:
        loaded = {arm: evaluation.load_fold_outputs(protocol, fold, arch_tag=suffix) for arm, suffix in arm_suffix.items()}
        reference = loaded[REFERENCE_ARM][1]
        for arm in arm_suffix:
            scale, outputs = loaded[arm]
            if not (np.array_equal(reference["labels"], outputs["labels"]) and np.array_equal(reference["group"], outputs["group"])):
                raise RuntimeError(f"fold {fold}: {arm} was not evaluated on the same test units, in the same order, as the patch graph")
            chosen_scale[arm][fold] = scale
            fold_outputs[arm].append(outputs)
    pooled = {arm: {key: np.concatenate([o[key] for o in outs]) for key in ("labels", "group", "appearance", "graph", "fused")} for arm, outs in fold_outputs.items()}
    labels, group = pooled[REFERENCE_ARM]["labels"], pooled[REFERENCE_ARM]["group"]
    clusters = evaluation.cluster_ids(group)

    lines = [f"# {cfg.DATASET}, {protocol}{cfg.ANCHOR_TAG}, folds {folds}: patch graph versus temporal-only control expert(s)", "",
             "Same units, folds and frozen appearance expert in every arm; only the Stage 2 expert differs. Chosen s per fold (validation only): "
             + "; ".join(f"{arm_name(arm)}: " + ", ".join(f"fold {f} {chosen_scale[arm][f]}" for f in folds) for arm in arm_suffix), "",
             f"Test units pooled: {len(labels)} (each unit once); clusters for the bootstrap: {len(set(clusters))}.", ""]
    rows = []
    for arm in arm_suffix:
        for variant in ("appearance", "graph", "fused"):
            top1, macro_f1 = run_utils.top1_and_macro_f1(labels, pooled[arm][variant])
            rows.append(dict(stage2_expert=arm_name(arm), variant="expert alone" if variant == "graph" else variant, top1=top1, macro_f1=macro_f1))
    lines += ["## Test performance (pooled)", "", "```", pd.DataFrame(rows).round(3).to_string(index=False), "```", ""]
    for control in controls:
        differ = int((pooled[REFERENCE_ARM]["appearance"] != pooled[control]["appearance"]).sum())
        lines.append(f"Units where the appearance expert of the {arm_name(control)} run predicts differently from the patch-graph run: {differ} of {len(labels)} "
                     "(must be 0: the same Stage 1 checkpoint is used).")
    lines.append("")

    confusions = {(arm, variant): cluster_confusions(labels, pooled[arm][variant], clusters) for arm in arm_suffix for variant in ("appearance", "graph", "fused")}
    table = paired_cluster_bootstrap(confusions, controls, cfg.N_BOOT, cfg.BOOT_SEED)
    lines += ["## Paired comparison, 95% bootstrap over clusters", "", "```", table.round(4).to_string(index=False), "```", "",
              "An interval of the difference above 0 means that the patch graph contributes beyond temporal access; a control gain of similar size with a difference around 0 "
              "means that the gain comes from the sequence input.", ""]
    out = cfg.REPORT_BASE / f"{cfg.DATASET}_{protocol}{cfg.ANCHOR_TAG}{cfg.PURITY_TAG}_graph_vs_temporal.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
