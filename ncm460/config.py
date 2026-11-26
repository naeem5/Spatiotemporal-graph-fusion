# every setting of the NCM460 experiments lives in this file.
import os
import sys
from pathlib import Path

# ======================================================================================================
# 1. PATHS
# ======================================================================================================
NCM_DIR = Path(__file__).resolve().parent                  # this folder
PROJECT_ROOT = NCM_DIR.parent                              # repository root
sys.path.insert(0, str(PROJECT_ROOT / "src"))              # the pdgraph library

# raw NCM460 images (TIFF, one file per sequence and channel):  <DATA_ROOT>/ncm460-p1/<channel>/Well..._Field.../<n>.tif
DATA_ROOT = Path(os.environ.get("NCM_DATA_ROOT", "data/ncm460"))   # folder that contains ncm460-p1/, ncm460-p2/, ...
LOCAL_ROOT = None                                          # copy of DATA_ROOT on a fast local disk, e.g. Path("ncm460_local");
                                                           # None = read the images from DATA_ROOT. Set it only after the copy has finished.
PATH_PREFIX_MAP = []                                       # (old prefix, new prefix) pairs applied to table paths; table paths are relative to DATA_ROOT

NCM_TABLE = NCM_DIR / "csv" / "NCM460_10class.csv"              # study table of NCM460, ten classes
# How NCM460 is divided into train / val / test:
#   "table": the split column of the study table, as for A549. In both tables every well contributes sequences to all three splits
#            (the split is random over cells), so this is the same protocol as the A549 results.
#   "field": new split in which every microscope field (all cells of one image position) belongs to one split only, so no field is shared
#            between training and test. Results are written to separate folders (runs_field_split, reports_field_split).
SPLIT_MODE = "table"
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)                       # train / val / test share of the sequences of each class (SPLIT_MODE = "field")
SPLIT_SEED = 2026
_SPLIT_SUFFIX = "" if SPLIT_MODE == "table" else "_field_split"
RUN_TAG = ""                                               # "" = the main runs; e.g. "_allframes" keeps a second set of runs (own runs/reports folders) next to them,
                                                           # for example Stage 1 trained on all frames. Stages listed in RUN_TAG_INHERITS are taken from the main runs.
RUN_TAG_INHERITS = ("stage2",)
_SUFFIX = _SPLIT_SUFFIX + RUN_TAG
NCM_INDEX_TABLE = NCM_DIR / "csv" / "ncm460_index_10class.csv"       # index with the split of the table (the field split is derived from it, without reading the images again)
NCM_INDEX = NCM_DIR / "csv" / f"ncm460_index_10class{_SPLIT_SUFFIX}.csv"       # written by 01_build_index.py (one row per sequence)
A549_TABLE = PROJECT_ROOT / "data" / "split_table.csv"                  # study table of A549 (for the cross-dataset tests)
A549_INDEX = NCM_DIR / "csv" / "a549_index_10class.csv"         # written by 01_build_index.py when BUILD_A549_INDEX is True
BUILD_A549_INDEX = False

BASE_RUNS_DIR = NCM_DIR / f"runs{_SPLIT_SUFFIX}"             # the main runs (RUN_TAG = "")
RUNS_DIR = NCM_DIR / f"runs{_SUFFIX}"                      # one folder per training run
REGISTRY = RUNS_DIR / "registry.json"                      # which run folder holds which stage/seed (written by the training scripts)
REPORT_DIR = NCM_DIR / f"reports{_SUFFIX}"                 # markdown/csv summaries


# ======================================================================================================
# 2. CLASSES (the same ten classes, in the same order, as in A549)
# ======================================================================================================
LABEL_COLUMN = "label"                                   # column of the NCM460 table that holds the class label
EXCLUDE_LABELS = ()                                        # labels of the table that are left out of every experiment (none)
CLASS_NAMES = ["Control", "Fission", "Fusion", "Interaction", "mtDNA", "Uncouplers", "Ionophores", "ETC", "ROS reduction", "ROS elevation"]
NUM_CLASSES = len(CLASS_NAMES)
CONTROL_LABEL = 0                                          # label of the untreated control (reference of the graph descriptors)

# ======================================================================================================
# 3. DATA
# ======================================================================================================
CHANNELS = ("mito", "mmp", "nuclear")                      # channel order of Stage 1 and Stage 2 (must not change after training)
TARGET_HW = 144                                            # frames are padded (bottom/right) or centre-cropped to TARGET_HW x TARGET_HW
Z_PLANES = 27                                              # Stage 1 needs exactly 27 z-planes; other stacks are resampled (see Z_MODE)
Z_MODE = "uniform"                                         # "uniform": pick evenly spaced planes (or repeat the nearest ones if there are fewer)
MIN_T = 3                                                  # sequences with fewer frames are not used by Stage 2/3 and the evaluation
GRAPH_Z_METHOD = "max"                                     # z-projection of the graph input
GRAPH_CACHE_DIR = NCM_DIR / "cache_graph"                  # z-projected sequences (about 14 GB), written once by 01e_cache_graph_inputs.py
GRAPH_CACHE_REQUIRED = True                                # True: Stage 2/3 stop if a sequence is not cached (instead of silently reading the slow drive)
CACHE_WORKERS = 3                                          # parallel readers while caching (few: the drive is slow at random access)
PROBE_WORKERS = 16                                       # threads that read the TIFF headers in 01_build_index.py

# ======================================================================================================
# 4. TRAINING (values of the A549 runs; see config.yaml of the A549 run folders)
# ======================================================================================================
SEEDS = [1234, 42, 2025]                                   # Stage 3 is trained once per seed on the same frozen experts
STAGE1_SEED = 1234
STAGE2_SEED = 1234
DEVICE_INDEX = 0                                           # GPU index (CPU is used when no GPU is available)
NUM_WORKERS = 8
STAGE1_READ_MODE = "middle_cache"                          # "middle_cache": one frame per sequence (the middle frame, the one Stage 3 uses), stored once on the local disk by
                                                           #                 01f_cache_stage1_frames.py (about 88 GB); an epoch then takes minutes
                                                           # "stream": all frames, read sequence by sequence from the slow disk
                                                           # "frame": all frames as random single frames (as in A549; needs the images on a fast local disk, see LOCAL_ROOT)
STAGE1_CACHE_DIR = NCM_DIR / "cache_stage1"                # written by 01f_cache_stage1_frames.py (table split only; the cache holds the train and val sequences)
STREAM_BUFFER = 200                                        # frames per worker in the shuffle buffer (uint16, about 0.7 GB per worker)
STAGE1_RESUME_RUN = None                                   # name of an unfinished Stage 1 folder in RUNS_DIR (e.g. "stage1_appearance_20261022-081500") to continue it
SKIP_FINISHED = True                                      # skip a training run that is already listed in the registry

STAGE1 = dict(epochs=100, batch_size=32, base_lr=3e-3, weight_decay=1e-3, warmup_epochs=5, min_lr=1e-6, embed_dim=256,
              weight_power=1.0, label_smoothing=0.1)
STAGE2 = dict(epochs=80, batch_size=16, base_lr=1e-3, weight_decay=1e-4, min_lr=1e-6, embed_dim=128, patch_stride=8, depth=2, num_heads=4,
              dropout=0.15, weight_power=0.5, label_smoothing=0.1, patience=15, grad_clip=5.0, selection_metric="macro_f1")
STAGE3 = dict(epochs=40, batch_size=16, lr=3e-4, weight_decay=1e-4, delta_scale=0.3, gate_tau=0.75, gate_alpha=10.0, gate_hidden=64,
              gate_dropout=0.1, prior_beta_init=0.5, weight_power=0.5, label_smoothing=0.1, lr_patience=3, patience=10,
              selection_metric="macro_f1")

# ======================================================================================================
# 5. EVALUATION
# ======================================================================================================
EVAL_SPLITS = ["val", "test"]                              # splits of NCM460 evaluated by 05_evaluate.py
EVAL_BATCH_SIZE = 16
FIXED_GATE = 1.0                                           # value of the constant-gate ablation
N_BOOT = 5000                                              # whole-well bootstrap replicates for the confidence intervals of the gain
BOOT_SEED = 2026                                           # same seed everywhere, so every table and figure reports the same interval
CAMERA_OFFSET = 100.0                                      # dark level of the camera; used by the optional brightness gain (off by default)
