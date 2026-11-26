# every setting of the public-dataset experiments (LCTP, HyperKvasir, PASTIS) lives in this file.
import json
import os
import sys
from pathlib import Path

PUB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PUB_DIR.parent / "src"))      # the pdgraph library

# ======================================================================================================
# 0. WHAT TO RUN
# ======================================================================================================
DATASET = os.environ.get("PUBLIC_DATASET", "lctp_a549")
IS_HK = DATASET == "hyperkvasir"                           # HyperKvasir labeled images: single RGB endoscopy images, NO time axis (see external_datasets/README.md)
IS_PASTIS = DATASET == "pastis"                            # PASTIS: Sentinel-2 satellite image time series of agricultural patches (see external_datasets/README.md)
IS_OFFICIAL_FOLDS = IS_HK or IS_PASTIS                     # data sets run with the official folds of the data set: protocol "official5", PUBLIC_FOLD 0-4
PROTOCOL = os.environ.get("PUBLIC_PROTOCOL", "official5" if IS_OFFICIAL_FOLDS else "plate")
FOLD = int(os.environ.get("PUBLIC_FOLD", "0"))
SMOKE = os.environ.get("PUBLIC_SMOKE", "0") == "1"
CELL_LINE = {"lctp_a549": "A549-WT", "lctp_u2os": "U2OS-WT"}.get(DATASET)
if DATASET not in ("lctp_a549", "lctp_u2os", "hyperkvasir", "pastis"):
    raise ValueError(f"PUBLIC_DATASET must be lctp_a549, lctp_u2os, hyperkvasir or pastis, not {DATASET!r}")
if PROTOCOL not in (("official5",) if IS_OFFICIAL_FOLDS else ("plate", "compound")):
    raise ValueError(f"protocol {PROTOCOL!r} does not exist for dataset {DATASET}")
# Reference of the Stage 2 descriptors. "classes" (default) = the reference classes the index builder wrote to the classes file (LCTP: DMSO; HyperKvasir: the three
# normal-looking landmark classes; PASTIS: meadow). "none" = no reference, the descriptors are used as they are (runs and reports get the suffix _noref).
REFERENCE = os.environ.get("PUBLIC_REFERENCE", "classes")
if REFERENCE not in ("classes", "none"):
    raise ValueError(f"PUBLIC_REFERENCE must be classes or none, not {REFERENCE!r}")
NO_REFERENCE = REFERENCE == "none"
HK_NORMAL_CLASSES = ["normal-cecum", "normal-pylorus", "normal-z-line"]
# Frames seen by the graph expert (Stage 2), LCTP and PASTIS only. "all" (default) = the whole time series. "anchor" = only the anchor frame, i.e. the same single frame the
# appearance expert sees (T = 1): the ablation that removes the time axis and nothing else. Runs and reports of "anchor" get the suffix _1frame.
FRAMES = os.environ.get("PUBLIC_FRAMES", "all")
if FRAMES not in ("all", "anchor") or (FRAMES == "anchor" and not (DATASET.startswith("lctp") or IS_PASTIS)):
    raise ValueError(f"PUBLIC_FRAMES must be all or anchor (anchor only for the LCTP and PASTIS data sets), not {FRAMES!r} for {DATASET}")
FRAMES_TAG = "_1frame" if FRAMES == "anchor" else ""
# Architecture of the Stage 2 expert. "graph" (default) = spatiotemporal patch graph. "temporal" = control expert with the same frames, per-frame CNN features, temporal attention and head,
# but the grid of each frame averaged to one vector (no spatial arrangement). "temporal_nodes" = grid kept, attention across time only (no spatial edges). The controls reuse the Stage 1
# run of the graph arm of the same fold; their runs and reports get the suffix _temporal or _temporal_nodes.
STAGE2_ARCH = os.environ.get("PUBLIC_STAGE2_ARCH", "graph")
if STAGE2_ARCH not in ("graph", "temporal", "temporal_nodes"):
    raise ValueError(f"PUBLIC_STAGE2_ARCH must be graph, temporal or temporal_nodes, not {STAGE2_ARCH!r}")
ARCH_TAG = "" if STAGE2_ARCH == "graph" else f"_{STAGE2_ARCH}"
# PASTIS only: the acquisition seen by the appearance expert (and, with PUBLIC_FRAMES=anchor, by the graph expert) is the one closest to a fixed calendar date:
# "jun" = 1 June 2019 (primary, peak vegetation for most French crops, chosen from agronomy and not from any result), "mar" = 15 March 2019, "sep" = 15 September 2019 (sensitivity).
PASTIS_ANCHOR = os.environ.get("PUBLIC_PASTIS_ANCHOR", "jun")
PASTIS_ANCHOR_DATES = {"jun": "2019-06-01", "mar": "2019-03-15", "sep": "2019-09-15"}
if PASTIS_ANCHOR not in PASTIS_ANCHOR_DATES:
    raise ValueError(f"PUBLIC_PASTIS_ANCHOR must be one of {list(PASTIS_ANCHOR_DATES)}, not {PASTIS_ANCHOR!r}")
ANCHOR_TAG = "" if (not IS_PASTIS or PASTIS_ANCHOR == "jun") else f"_{PASTIS_ANCHOR}"
# PASTIS only, secondary analysis decided before any model result: with PUBLIC_MIN_PURITY=0.6 the reports (09, 10, 11) use only the test patches whose dominant crop covers at least
# 60% of the crop pixels (the patch label is cleaner there). Training is unchanged; only the test units that are reported are selected. Report names get the suffix _pur60.
MIN_PURITY = float(os.environ.get("PUBLIC_MIN_PURITY", "0"))
if MIN_PURITY > 0 and not IS_PASTIS:
    raise ValueError("PUBLIC_MIN_PURITY is only defined for PASTIS")
PURITY_TAG = f"_pur{round(100 * MIN_PURITY)}" if MIN_PURITY > 0 else ""
# PASTIS official folds are numbered 1-5; run k (PUBLIC_FOLD 0-4) tests fold k+1, validates on the next fold (5 -> 1) and trains on the other three.
PASTIS_TEST_FOLD = FOLD + 1
PASTIS_VAL_FOLD = (FOLD + 1) % 5 + 1
PASTIS_TRAIN_FOLDS = [f for f in range(1, 6) if f not in (PASTIS_TEST_FOLD, PASTIS_VAL_FOLD)]

# ======================================================================================================
# 1. PATHS
# ======================================================================================================
# LCTP: the folder that holds A549-WT.zip (40.0 GiB) and U2OS-WT.zip (66.8 GiB), or the already unzipped folders A549-WT/ and U2OS-WT/.
LCTP_ROOT = Path(os.environ.get("PUBLIC_LCTP_ROOT", "data/lctp"))

# HyperKvasir: the folder that holds hyper-kvasir-labeled-images.zip (3.93 GB), or the already unzipped folder labeled-images/. The official split file is
# csv/hyperkvasir_5_fold_split.csv (copy of official_splits/5_fold_split.csv of github.com/simula/hyper-kvasir). Images are cached once as uint8 (3, 144, 144).
HK_ROOT = Path(os.environ.get("PUBLIC_HK_ROOT", "data/HyperKvasir"))
HK_SPLIT_FILE = PUB_DIR / "csv" / "hyperkvasir_5_fold_split.csv"

# PASTIS: the UNZIPPED PASTIS.zip (28.8 GB; Zenodo 10.5281/zenodo.5012942, CC BY 4.0): the folder that contains metadata.geojson, DATA_S2/, ANNOTATIONS/ and NORM_S2_patch.json.
PASTIS_ROOT = Path(os.environ.get("PUBLIC_PASTIS_ROOT", "data/PASTIS"))
HK_CACHE = Path(os.environ.get("PUBLIC_TILE_CACHE", str(PUB_DIR / "cache_tiles"))) / "hyperkvasir" / "images_144.npy"

CSV_DIR = Path(os.environ.get("PUBLIC_CSV_DIR", str(PUB_DIR / "csv")))   # PUBLIC_CSV_DIR / PUBLIC_OUT_DIR redirect the outputs (code tests)
INDEX = CSV_DIR / f"{DATASET}_index.csv"                   # one row per tile (LCTP), image (HyperKvasir) or patch (PASTIS); all splits of all protocols are columns
CLASSES_JSON = CSV_DIR / f"{DATASET}_classes.json"         # class names, control class, counts
PLATE_MAP = PUB_DIR / "csv" / "lctp_plate_map.csv"         # well -> compound, mechanism group (from the LCTP data repository)
OUT_DIR = Path(os.environ.get("PUBLIC_OUT_DIR", str(PUB_DIR)))
TILE_CACHE = Path(os.environ.get("PUBLIC_TILE_CACHE", str(PUB_DIR / "cache_tiles"))) / (CELL_LINE or "none")   # uint8 tile sequences (about 0.3 MB each)
REPORT_BASE = OUT_DIR / "reports"
_TAG_CORE = f"{DATASET}_{PROTOCOL}" + f"_f{FOLD}" + ("_noref" if NO_REFERENCE else "") + ANCHOR_TAG + FRAMES_TAG
RUN_TAG = _TAG_CORE + ARCH_TAG + ("_smoke" if SMOKE else "")
BASE_RUN_TAG = _TAG_CORE + ("_smoke" if SMOKE else "")       # graph-arm run of the same fold: the temporal controls reuse its Stage 1
PASTIS_NORM_FILE = CSV_DIR / "pastis_norm.json"            # per-channel mean and std of every official fold (copy of NORM_S2_patch.json, written by 03_build_index_pastis.py)
RUNS_DIR = OUT_DIR / "runs" / RUN_TAG
BASE_RUNS_DIR = OUT_DIR / "runs" / BASE_RUN_TAG
REPORT_DIR = REPORT_BASE / RUN_TAG
REGISTRY = RUNS_DIR / "registry.json"
FEATURE_DIR = RUNS_DIR / "stage3_features"

SPLIT_COLUMN = f"split_official5_f{FOLD}" if IS_OFFICIAL_FOLDS else (f"split_plate_f{FOLD}" if PROTOCOL == "plate" else f"split_compound_f{FOLD}")
SPLIT_ALIASES = {"test": ["test", "test_query", "test_gallery"]}   # asking for "test" returns every test part of the protocol


def load_classes():
    # {'names': [...], 'control_label': int, ...} written by the index builder.
    if not CLASSES_JSON.exists():
        raise FileNotFoundError(f"{CLASSES_JSON} is missing: run the index builder first (01_build_index_lctp.py, 02_build_index_hyperkvasir.py or 03_build_index_pastis.py)")
    return json.loads(CLASSES_JSON.read_text(encoding="utf-8"))


def class_names():
    return load_classes()["names"]


def num_classes():
    return len(class_names())


def control_label():
    return int(load_classes()["control_label"])


def control_labels():
    # Labels whose sequences build the reference of the Stage 2 descriptors: [DMSO] for LCTP; the reference classes written by the index builder for
    if NO_REFERENCE:
        return []
    classes = load_classes()
    return [int(v) for v in classes["control_labels"]] if "control_labels" in classes else [control_label()]


# ======================================================================================================
# 2. DATA
# ======================================================================================================
CHANNELS = ("red", "green", "blue") if IS_HK else tuple(f"s2_band{i}" for i in range(1, 11)) if IS_PASTIS else ("phase",)
                                                           # HyperKvasir: RGB. PASTIS: the 10 Sentinel-2 bands of the data set (only the number is used).
                                                           # LCTP: phase contrast only.
TARGET_HW = 144                                            # frames are padded (bottom/right, zeros) or centre-cropped to 144 x 144, as in the A549 experiments
Z_PLANES = 1                                               # neither data set has a z-stack: the appearance expert sees one focal plane (2D variant)
MIN_T = 1 if IS_HK else 3                                  # HyperKvasir: one image = a "sequence" of one frame (T = 1); every temporal descriptor is then inert (persistence = 1)
GRAPH_NORM = dict(q_low=0.01, q_high=0.99)                 # Stage 2 input: log1p, then 1st-99th percentile scaling per channel and sequence (as A549)
GRAPH_INPUT = "unit" if IS_HK else "standardize" if IS_PASTIS else "robust"              # HyperKvasir: uint8 / 255 for both experts. The per-image, per-channel percentile stretch of "robust" would remove the colour
                                                           # cast of the mucosa and of the blue dye (dyed-lifted-polyps), and these images are well exposed.
                                                           # PASTIS: "standardize" for both experts = per-channel (value - mean) / std with the mean and std of the TRAINING folds (as the data set authors' loader), no per-image rescaling.
STAGE1_INPUT = "unit" if IS_HK else "standardize" if IS_PASTIS else "robust"                                # Stage 1 input: "robust" = log1p + 1st-99th percentile scaling per channel and frame (as Stage 2); "unit" = uint8 / 255.

# --- LCTP (Forsgren et al.) ---
TILE = 144                                                 # tile edge in pixels (native resolution, 1040 x 1408 -> 7 x 9 tiles)
WINDOW_HOURS = list(range(0, 73, 6))                       # frames used by the graph expert: the scans nearest to these hours after the first scan (13 frames)
ANCHOR_HOUR = 48                                           # frame seen by the appearance expert; 48 h is the single time point of the LCTP paper
MAX_SCAN_DEVIATION_H = 1.5                                 # a scan further than this from a wanted hour is not used
TILES_PER_WELL = 8
TILES_PER_DMSO_WELL = 32                                   # DMSO has 15 wells per plate against 60 per class, so more tiles per well keeps the classes balanced
MIN_CELLS_PER_TILE = 2                                     # tiles with fewer cells at the anchor frame (AichPhLabels mask) are not drawn
TILE_SEED = 2026
VAL_COMPOUNDS_PER_CLASS = 2                                # of the 12 compounds of a class
VAL_DMSO_WELLS_PER_PLATE = 2
COMPOUND_FOLDS = 4                                         # compound protocol: 3 test compounds per class and fold
BUILD_WORKERS = int(os.environ.get("PUBLIC_BUILD_WORKERS", "8"))
LCTP_CLASS_ORDER = ["DMSO", "AKT", "CDK", "HDAC", "MAPK", "PARP", "TUB"]    # label 0 = DMSO = control

# ======================================================================================================
# 3. TRAINING (values of the A549 runs)
# ======================================================================================================
SEEDS = [1234, 42, 2025]                                   # Stage 3 is trained once per seed on the same frozen experts
STAGE1_SEED = 1234
STAGE2_SEED = 1234
DEVICE_INDEX = int(os.environ.get("PUBLIC_GPU", "0"))
NUM_WORKERS = int(os.environ.get("PUBLIC_WORKERS", "8"))
SKIP_FINISHED = os.environ.get("PUBLIC_RETRAIN", "0") != "1"      # a stage already registered in runs/<tag>/registry.json is skipped; PUBLIC_RETRAIN=1 trains it again

# Stage 1 network: torchvision ResNet-18 from scratch with a linear head (the data have no z-stack).
# Adam lr = base_lr x batch / 256 = 1e-3, plain cross-entropy (class weights and label smoothing slowed the start), random flips and
# 90-degree rotations of the frame. The checkpoint with the best validation top-1 is kept (the network overfits a training set of this size within ~15 epochs).
STAGE1 = dict(epochs=40, batch_size=32, base_lr=8e-3, weight_decay=1e-4, warmup_epochs=3, min_lr=1e-6, embed_dim=256,
              weight_power=0.0, label_smoothing=0.0, augment=True)
STAGE2 = dict(epochs=80, batch_size=16, base_lr=1e-3, weight_decay=1e-4, min_lr=1e-6, embed_dim=128, patch_stride=8, depth=2, num_heads=4,
              dropout=0.15, weight_power=0.5, label_smoothing=0.1, patience=15, grad_clip=5.0, selection_metric="macro_f1")
if IS_PASTIS:                                              # up to 61 dates x 10 bands per sample: half the batch of the other data sets (memory; technical change only)
    STAGE2 = {**STAGE2, "batch_size": 8}
if "PUBLIC_STAGE2_BATCH" in os.environ:                    # e.g. PUBLIC_STAGE2_BATCH=4 if the GPU runs out of memory
    STAGE2 = {**STAGE2, "batch_size": int(os.environ["PUBLIC_STAGE2_BATCH"])}
STAGE3 = dict(epochs=40,batch_size=16, lr=3e-4, weight_decay=1e-4, delta_scale=0.3, gate_tau=0.75, gate_alpha=10.0, gate_hidden=64,
              gate_dropout=0.1, prior_beta_init=0.5, weight_power=0.5, label_smoothing=0.1, lr_patience=3, patience=10,
              selection_metric="macro_f1")
REFERENCE_SCALE = 0.3                                      # s of the A549 runs, reported next to the s chosen on the validation set
GATE_SCALE_GRID = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# ======================================================================================================
# 4. EVALUATION
# ======================================================================================================
EVAL_BATCH_SIZE = 16
N_BOOT = 5000
BOOT_SEED = 2026

if SMOKE:                                                  # tiny run to test the code on a few sequences
    STAGE1 = {**STAGE1, "epochs": 1, "batch_size": 8, "warmup_epochs": 1}
    STAGE2 = {**STAGE2, "epochs": 1, "batch_size": 4}
    STAGE3 = {**STAGE3, "epochs": 2, "batch_size": 4}
    GATE_SCALE_GRID = [0.3, 1.0]
    SEEDS = [1234]
    NUM_WORKERS = 0
    N_BOOT = 50
    EVAL_BATCH_SIZE = 4
