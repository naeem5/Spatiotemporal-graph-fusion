# run a complete Stage 3 model on a set of sequences and summarise the result (the shared parts are in pdgraph.evaluation)
import numpy as np
import pandas as pd

import config as cfg
import data_loading
import run_utils
from pdgraph import evaluation as shared
from pdgraph.evaluation import FEATURE_KEYS, average_gates as ensemble_outputs  # noqa: F401  (used by the scripts)
from pdgraph.metrics import exact_mcnemar, group_bootstrap_gain  # noqa: F401


def extract_features(model, records, device, batch_size=None, workers=None, gain=None):
    # run the two frozen experts of a Stage 3 model on the records (the only pass that reads the images)
    loader = data_loading.make_loader(data_loading.FusionDataset(records, gain), batch_size or cfg.EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
                                      collate_fn=data_loading.collate_fusion, workers=workers)
    result = shared.expert_features(model, loader, device)
    result["row_id"] = np.array([int(r["row_id"]) for r in records])
    result["graph_path"] = np.array([r["path_mito"] for r in records], dtype=object)
    result["well"] = np.array([f'{r["batch"]}|{r["well"]}' for r in records], dtype=object)       # lets the bootstrap resample whole wells
    return result


def cached_features(split, records, model, device):
    # expert outputs of an NCM460 split, stored in runs/stage3_features/<split>.npz
    meta = {"stage1": run_utils.registered_run("stage1").name, "stage2": run_utils.registered_run("stage2").name, "min_T": cfg.MIN_T}
    rows = np.array([int(r["row_id"]) for r in records])
    return shared.load_or_compute_features(cfg.RUNS_DIR / "stage3_features", split, rows, meta, lambda: extract_features(model, records, device))


def fused_outputs(model, features, device, batch_size=1024):
    # gate and fusion of a Stage 3 model on stored expert outputs; the arrays of test_features.npz (plus 'well')
    out = shared.fused_outputs(model, features, device, batch_size)
    out["graph_path"], out["well"] = features["graph_path"], features["well"]
    return out


def well_bootstrap_gain(labels, base_pred, new_pred, wells, n_boot=None, seed=None):
    # gain of new_pred over base_pred with a 95% percentile interval from a whole-well bootstrap
    result = group_bootstrap_gain(labels, base_pred, new_pred, wells, cfg.NUM_CLASSES, n_boot or cfg.N_BOOT, cfg.BOOT_SEED if seed is None else seed)
    result["n_wells"] = result.pop("n_groups")
    return result


def summary_table(out):
    # one row per variant: top-1 (%), macro-F1, and the sequences corrected / broken compared with Stage 1 alone
    return shared.variant_table(out["labels"], prediction_variants(out))


def collect_outputs(model, records, device, batch_size=None, workers=None, gain=None):
    # Read the records, run the experts and the gate of the Stage 3 model; returns the arrays of test_features.npz (plus 'well').
    return fused_outputs(model, extract_features(model, records, device, batch_size, workers, gain), device)

def prediction_variants(out):
    # Predictions of Stage 1 alone, Stage 2 alone and the four gate variants (constant gate, prior only, learned only, full gate).
    s = cfg.STAGE3["delta_scale"]
    logits = {
        "appearance": out["logits_app"],
        "graph": out["delta_logits"],
        f"fixed({cfg.FIXED_GATE:g})": out["logits_app"] + s * cfg.FIXED_GATE * out["delta_logits"],
        "prior_only": out["logits_app"] + s * out["prior_gate"] * out["delta_logits"],
        "learned_only": out["logits_app"] + s * out["learned_gate"] * out["delta_logits"],
        "full_gate": out["logits_app"] + s * out["gate"] * out["delta_logits"],
    }
    return {name: z.argmax(axis=1) for name, z in logits.items()}


def per_class_recall(out, variant_preds):
    # Recall (%) of every class for each variant, as a DataFrame (rows = classes).
    labels = out["labels"]
    table = {}
    for name, p in variant_preds.items():
        table[name] = [100.0 * float((p[labels == c] == c).mean()) if (labels == c).any() else np.nan for c in range(cfg.NUM_CLASSES)]
    df = pd.DataFrame(table, index=cfg.CLASS_NAMES)
    df["n"] = [int((labels == c).sum()) for c in range(cfg.NUM_CLASSES)]
    return df


def save_arrays(out, path):
    # Save the outputs and the predictions of every variant (keys as in test_features.npz, plus 'well').
    items = dict(out)
    for name, p in prediction_variants(out).items():
        items["preds_" + name.replace("(", "_").replace(")", "").replace(".", "_")] = p
    np.savez(path, **items)
