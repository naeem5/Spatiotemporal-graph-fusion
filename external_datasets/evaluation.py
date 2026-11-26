# run the frozen experts and the gate on a set of recordings and summarise the result (the shared parts are in pdgraph.evaluation)
import numpy as np
import pandas as pd

import config as cfg
import data_loading
import run_utils
from pdgraph import evaluation as shared
from pdgraph.evaluation import FEATURE_KEYS, average_gates as ensemble_outputs  # noqa: F401  (used by the scripts)
from pdgraph.metrics import exact_mcnemar, group_bootstrap_gain as bootstrap_gain  # noqa: F401


def extract_features(model, records, device, batch_size=None, workers=None):
    # run the two frozen experts of a Stage 3 model on the records (the only pass that reads the images)
    loader = data_loading.make_loader(data_loading.FusionDataset(records), batch_size or cfg.EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
                                      collate_fn=data_loading.collate_fusion, workers=workers)
    result = shared.expert_features(model, loader, device)
    result["row_id"] = np.array([int(r["row_id"]) for r in records])
    result["group"] = np.array([str(r["well"]) for r in records], dtype=object)         # whole groups are resampled by the bootstrap
    result["part"] = np.array([str(r["split"]) for r in records], dtype=object)
    return result


def cached_features(split, records, model, device):
    # expert outputs of a split, stored in runs/<tag>/stage3_features/<split>.npz (reused only if Stage 1/2 and the rows are the same)
    meta = {"stage1": run_utils.registered_run("stage1").name, "stage2": run_utils.registered_run("stage2").name, "min_T": cfg.MIN_T}
    rows = np.array([int(r["row_id"]) for r in records])
    return shared.load_or_compute_features(cfg.FEATURE_DIR, split, rows, meta, lambda: extract_features(model, records, device))


def fused_outputs(model, features, device, batch_size=1024):
    # gate and fusion of a Stage 3 model on stored expert outputs
    out = shared.fused_outputs(model, features, device, batch_size)
    out["group"], out["part"], out["row_id"] = features["group"], features["part"], features["row_id"]
    return out


def group_bootstrap_gain(labels, base_pred, new_pred, groups, n_boot=None, seed=None):
    return bootstrap_gain(labels, base_pred, new_pred, groups, cfg.num_classes(), n_boot or cfg.N_BOOT, cfg.BOOT_SEED if seed is None else seed)


def summary_table(out, scale):
    return shared.variant_table(out["labels"], prediction_variants(out, scale), with_count=True)


def prediction_variants(out, scale):
    # Predictions of the appearance expert alone, the graph expert alone and the gated fusion at strength `scale`.
    z = {"appearance": out["logits_app"], "graph": out["delta_logits"], "fused": out["logits_app"] + scale * out["gate"] * out["delta_logits"]}
    return {name: v.argmax(axis=1) for name, v in z.items()}


def cluster_ids(group):
    # Ids of the units that are resampled together in a bootstrap, for the stored 'group' of the test units: LCTP = compound (all its tiles, wells and plates; every DMSO
    if not cfg.DATASET.startswith("lctp"):
        return np.asarray(group).astype(str)
    idx = pd.read_csv(cfg.INDEX)
    well_to_compound = dict(idx[["well", "compound"]].drop_duplicates().itertuples(index=False))
    control = cfg.load_classes()["names"][cfg.load_classes()["control_label"]]
    control_wells = set(idx.loc[idx["moa"] == control, "well"])
    return np.array([w if w in control_wells else well_to_compound[w] for w in group])


def _stored_fold(protocol, fold, frames_tag=None, arch_tag=None):
    # (s chosen on the validation set, the stored test arrays of 07_evaluate.py with their prefix removed) of one fold.
    # arch_tag: "" (patch graph), "_temporal" or "_temporal_nodes"; default = the active setting.
    import re
    tag = cfg.FRAMES_TAG if frames_tag is None else frames_tag
    arch = cfg.ARCH_TAG if arch_tag is None else arch_tag
    suffix = ("_noref" if cfg.NO_REFERENCE else "") + cfg.ANCHOR_TAG                  # the same order as RUN_TAG in config.py
    folder = cfg.REPORT_BASE / f"{cfg.DATASET}_{protocol}_f{fold}{suffix}{tag}{arch}"
    scale = float(re.search(r"Chosen s = ([0-9.]+)", (folder / "results.md").read_text(encoding="utf-8")).group(1))
    z = np.load(folder / "test_outputs.npz", allow_pickle=True)
    return scale, (lambda key: z[f"s{scale}_{key}"])


def _kept_units(group):
    # Boolean mask of the test units that enter the report: all of them, except PASTIS with PUBLIC_MIN_PURITY > 0, where the patches whose dominant crop covers less of the crop
    if cfg.MIN_PURITY <= 0:
        return np.ones(len(group), dtype=bool)
    idx = pd.read_csv(cfg.INDEX)
    purity = dict(zip(idx["well"], idx["purity"]))
    return np.array([purity[g] >= cfg.MIN_PURITY for g in group])


def load_fold_outputs(protocol, fold, frames_tag=None, arch_tag=None):
    # Stored test result of one fold (written by 07_evaluate.py) at the strength s that fold chose on its validation set.
    scale, get = _stored_fold(protocol, fold, frames_tag, arch_tag)
    d = dict(labels=get("labels"), group=get("group").astype(str), appearance=get("preds_appearance"), graph=get("preds_graph"), fused=get("preds_fused"))
    keep = _kept_units(d["group"])
    return scale, {k: v[keep] for k, v in d.items()}


def load_fold_scores(protocol, fold, frames_tag=None, arch_tag=None):
    # Like load_fold_outputs, but with the class scores (logits) instead of the predictions, for top-k accuracy: appearance, graph (graph logits minus their mean over
    scale, get = _stored_fold(protocol, fold, frames_tag, arch_tag)
    app, delta = get("logits_app"), get("delta_logits")
    d = dict(labels=get("labels"), group=get("group").astype(str), appearance=app, graph=delta, fused=app + scale * get("gate") * delta)
    keep = _kept_units(d["group"])
    return scale, {k: v[keep] for k, v in d.items()}

