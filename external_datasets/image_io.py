# reading the images of the public data sets and putting them into the shape the models expect.
import json
from pathlib import Path

import numpy as np

import config as cfg
from pdgraph.data import fit_hw as fit_hw_to_size
from pdgraph.data import robust_norm as robust_norm_with_quantiles


def fit_hw(arr, target=None):
    return fit_hw_to_size(arr, target or cfg.TARGET_HW)


def robust_norm(arr):
    # log1p, then clip to the quantiles of cfg.GRAPH_NORM and scale to [0, 1]; one array = one channel of one sequence
    return robust_norm_with_quantiles(arr, cfg.GRAPH_NORM["q_low"], cfg.GRAPH_NORM["q_high"])


def graph_input(seq_u8):
    # (C, T, H, W) uint8 -> float32 Stage 2 input (log1p + percentile scaling per channel).
    if cfg.GRAPH_INPUT == "standardize":                        # PASTIS: already float32 and standardised by load_sequence
        return np.ascontiguousarray(seq_u8, dtype=np.float32)
    if cfg.GRAPH_INPUT == "unit":
        return np.ascontiguousarray(seq_u8.astype(np.float32) / 255.0)
    return np.ascontiguousarray(np.stack([robust_norm(ch) for ch in seq_u8]).astype(np.float32))


def appearance_input(frame_u8):
    # (C, H, W) uint8 -> float32 (C, Z=1, H, W) Stage 1 input in [0, 1].
    if cfg.STAGE1_INPUT == "standardize":                       # PASTIS: already float32 and standardised by load_frame
        return np.ascontiguousarray(frame_u8, dtype=np.float32)[:, None]
    if cfg.STAGE1_INPUT == "robust":
        return np.ascontiguousarray(np.stack([robust_norm(ch) for ch in frame_u8]).astype(np.float32)[:, None])
    return np.ascontiguousarray((frame_u8.astype(np.float32) / 255.0)[:, None])


# ------------------------------------------------------------------------------------------------------
# generic access used by the datasets
# ------------------------------------------------------------------------------------------------------
_HK_IMAGES = None


def _hk_image(rec):
    # HyperKvasir image of a record as (3, 144, 144) uint8 from the cache written by 02_build_index_hyperkvasir.py (opened once per process, memory-mapped).
    global _HK_IMAGES
    if _HK_IMAGES is None:
        _HK_IMAGES = np.load(cfg.HK_CACHE, mmap_mode="r")
    return np.ascontiguousarray(_HK_IMAGES[int(rec["cache_row"])])


_PASTIS_STATS = None


def _pastis_standardise(arr):
    # (C, ...) float32 -> (value - mean) / std per channel, with the mean and std averaged over the TRAINING folds of the active run (as the loader of the data set
    global _PASTIS_STATS
    if _PASTIS_STATS is None:
        norm = json.loads(cfg.PASTIS_NORM_FILE.read_text(encoding="utf-8"))
        mean = np.mean([norm[f"Fold_{f}"]["mean"] for f in cfg.PASTIS_TRAIN_FOLDS], axis=0)
        std = np.mean([norm[f"Fold_{f}"]["std"] for f in cfg.PASTIS_TRAIN_FOLDS], axis=0)
        _PASTIS_STATS = (mean.astype(np.float32), std.astype(np.float32))
    shape = (-1,) + (1,) * (arr.ndim - 1)
    return (arr - _PASTIS_STATS[0].reshape(shape)) / _PASTIS_STATS[1].reshape(shape)


def _pastis_series(rec):
    # Memory-mapped (T, C, H, W) array of a PASTIS patch (as stored in DATA_S2/S2_<id>.npy).
    return np.load(cfg.PASTIS_ROOT / "DATA_S2" / f"S2_{int(rec['patch_id'])}.npy", mmap_mode="r")


def load_frame(rec, t):
    # Frame t (0-based) of the sequence as (C, H, W) uint8 (PASTIS: float32, standardised), padded to cfg.TARGET_HW.
    if cfg.IS_HK:
        return _hk_image(rec)
    if cfg.IS_PASTIS:
        return fit_hw(_pastis_standardise(np.asarray(_pastis_series(rec)[t], dtype=np.float32)))      # reads only this acquisition
    tile = np.load(Path(rec["path"]) if Path(rec["path"]).is_absolute() else cfg.PUB_DIR / rec["path"], mmap_mode="r")      # (T, H, W)
    return fit_hw(np.ascontiguousarray(tile[t])[None])


def load_sequence(rec):
    # Whole sequence as (C, T, H, W) uint8 (PASTIS: float32, standardised), padded to cfg.TARGET_HW.
    if cfg.IS_HK:
        return _hk_image(rec)[:, None]                          # one image = one frame
    if cfg.IS_PASTIS:
        series = _pastis_series(rec)                            # (T, C, H, W)
        if cfg.FRAMES == "anchor":                              # ablation: the graph expert sees only the anchor acquisition, T = 1
            series = series[min(anchor_index(rec), series.shape[0] - 1):][:1]
        return fit_hw(_pastis_standardise(np.asarray(series, dtype=np.float32).transpose(1, 0, 2, 3)))
    tile = np.load(Path(rec["path"]) if Path(rec["path"]).is_absolute() else cfg.PUB_DIR / rec["path"])      # (T, H, W)
    if cfg.FRAMES == "anchor":                                  # ablation: the graph expert sees only the anchor (48 h) frame, T = 1
        a = min(int(rec["anchor"]), tile.shape[0] - 1)
        tile = tile[a:a + 1]
    return fit_hw(tile[None])


def anchor_index(rec):
    # Index of the frame the appearance expert sees: the 48 h frame for LCTP, the acquisition closest to the anchor
    if cfg.IS_HK:
        return 0
    if cfg.IS_PASTIS:
        return int(rec[f"anchor_{cfg.PASTIS_ANCHOR}"])
    return int(rec["anchor"])
