# the study index and the three datasets (Stage 1, Stage 2, Stage 3) for NCM460 and A549.
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm import tqdm

import config as cfg
import image_io
from pdgraph.data import class_weights as class_weights_from_counts
from pdgraph.data import collate_fusion, collate_graph  # noqa: F401  (re-exported for the scripts)
from pdgraph.data import make_loader as make_seeded_loader
from pdgraph.utils import set_seed  # noqa: F401  (re-exported for the scripts)

GROUP_KEYS = ["batch", "well", "field", "filename"]            # these four columns identify one sequence in the study tables


# ======================================================================================================
# study index
# ======================================================================================================
def build_index(table_csv, label_column, exclude_labels=(), dataset_name="ncm460", workers=None):
    # Read a study table (one row per sequence AND channel) and return the index (one row per sequence).
    table = pd.read_csv(table_csv)
    table = table[table["channel"].isin(cfg.CHANNELS)]
    if exclude_labels:
        table = table[~table[label_column].isin(exclude_labels)]
    rows = []
    for key, g in table.groupby(GROUP_KEYS, sort=False):
        g = g.set_index("channel")
        if any(c not in g.index for c in cfg.CHANNELS):
            continue
        first = g.iloc[0]
        rec = dict(dataset=dataset_name, split=first["split"], batch=key[0], well=key[1], field=key[2], filename=key[3],
                   label=int(first[label_column]), group=first.get("group", ""), name=first.get("name", ""))
        for c in cfg.CHANNELS:
            rec[f"path_{c}"] = str(image_io.resolve_path(g.loc[c, "path"]))
        rows.append(rec)
    index = pd.DataFrame(rows)

    def probe_sequence(rec):
        shapes = [image_io.probe(rec[f"path_{c}"]) for c in cfg.CHANNELS]
        if any(s is None for s in shapes) or len({s[:4] for s in shapes}) != 1:
            return (np.nan, np.nan, np.nan, np.nan, "", False)
        return (*shapes[0][:4], shapes[0][4], True)

    with ThreadPoolExecutor(workers or cfg.PROBE_WORKERS) as ex:
        results = list(tqdm(ex.map(probe_sequence, rows), total=len(rows), desc=f"reading headers ({dataset_name})"))
    index[["T", "Z", "H", "W", "dtype", "ok"]] = pd.DataFrame(results, index=index.index)
    for c in ("T", "Z", "H", "W"):
        index[c] = index[c].astype("Int64")
    return index


COMPOUND_ALIASES = {"Mitoquinone mesylate": "MitoQ", "Ciclosporin A": "Cyclosporin A", "Oligomycin": "Oligomycin A"}   # A549 spelling -> NCM460 spelling


def canonical_name(name):
    # Compound name spelled the same way in both datasets.
    name = str(name).strip()
    return COMPOUND_ALIASES.get(name, name)


def compound_scheme(index_csv):
    # {compound: class label} of an index file. A compound must have exactly one class inside a dataset.
    df = pd.read_csv(index_csv)
    labels = df[df["ok"].astype(bool)].groupby("name")["label"].agg(lambda s: sorted(set(int(v) for v in s)))
    if (labels.str.len() != 1).any():
        raise ValueError(f"some compounds have more than one class in {index_csv}: {labels[labels.str.len() != 1].to_dict()}")
    return {canonical_name(n): v[0] for n, v in labels.items()}


def relabel_to_scheme(records, scheme):
    # Copy of the records with the class label taken from `scheme` (compound -> class); returns (records, compounds missing from the scheme).
    out, missing = [], set()
    for r in records:
        key = canonical_name(r["name"])
        if key in scheme:
            out.append({**r, "label": scheme[key]})
        else:
            missing.add(key)
    return out, missing


def resplit_by_field(index, fractions, seed):
    # New train/val/test split in which each microscope field (batch, well, field) belongs to exactly one split.
    index = index.copy()
    rng = np.random.default_rng(seed)
    cut = np.cumsum(fractions)[:-1]
    index["split"] = ""
    for label, g in index.groupby("label"):
        fields = g.groupby(["batch", "well", "field"]).size()
        order = rng.permutation(len(fields))
        share = np.cumsum(fields.values[order]) / fields.sum()
        assigned = np.searchsorted(cut, share - 1e-9, side="left")        # 0 = train, 1 = val, 2 = test
        for key, k in zip(fields.index[order], assigned):
            sel = (index["label"] == label) & (index["batch"] == key[0]) & (index["well"] == key[1]) & (index["field"] == key[2])
            index.loc[sel, "split"] = ("train", "val", "test")[int(k)]
    return index


def load_index(csv_path, split=None, min_T=1, to_local=True):
    # Records (list of dicts) of the sequences that are readable, have at least min_T frames and belong to the split ('all' = every split).
    df = pd.read_csv(csv_path)
    df["row_id"] = df.index                                    # position in the index file: names the cache file of the sequence
    df = df[df["ok"].astype(bool)]
    if split not in (None, "all"):
        df = df[df["split"] == split]
    df = df[df["T"] >= min_T]
    records = df.to_dict("records")
    if to_local and cfg.LOCAL_ROOT is not None:
        old, new = str(cfg.DATA_ROOT), str(cfg.LOCAL_ROOT)
        for r in records:
            for c in cfg.CHANNELS:
                if str(r[f"path_{c}"]).startswith(old):
                    r[f"path_{c}"] = new + str(r[f"path_{c}"])[len(old):]
    return records


# ======================================================================================================
# batching helpers
# ======================================================================================================


# ======================================================================================================
# Stage 1: appearance expert (one frame, full z-stack)
# ======================================================================================================
def _frame_stack(rec, t):
    # Frame t of the three channels as (3, Z, H, W) float32 with exactly cfg.Z_PLANES planes and cfg.TARGET_HW x cfg.TARGET_HW pixels.
    planes = [image_io.match_z(image_io.read_frame(rec[f"path_{c}"], t, int(rec["Z"]))) for c in cfg.CHANNELS]
    return np.ascontiguousarray(image_io.fit_hw(np.stack(planes).astype(np.float32)))


def calibrate(arr, gain):
    # Multiply the signal above the camera offset by a gain per channel (axis 0); pixels that are zero (outside the cell) stay zero.
    if gain is None:
        return arr
    g = np.asarray(gain, dtype=np.float32).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = cfg.CAMERA_OFFSET + (arr.astype(np.float32) - cfg.CAMERA_OFFSET) * g
    return np.where(arr > 0, np.maximum(out, 0.0), 0.0).astype(np.float32)


def class_weights(records, power):
    # inverse-frequency class weights from the training sequences; every class must be present
    counts = np.bincount([r["label"] for r in records], minlength=cfg.NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(f"the training split has no sequences of the classes {np.where(counts == 0)[0].tolist()}")
    return class_weights_from_counts(counts, power)


def make_loader(dataset, batch_size, shuffle, drop_last, collate_fn=None, seed=0, workers=None):
    return make_seeded_loader(dataset, batch_size, shuffle, drop_last, collate_fn, seed, cfg.NUM_WORKERS if workers is None else workers)


class AppearanceFrameDataset(Dataset):
    # Stage 1 sample = one frame of one sequence with raw intensities (no normalisation, exactly as in the A549 Stage 1 runs).

    def __init__(self, records):
        self.records = records
        self.items = [(i, t) for i, r in enumerate(records) for t in range(int(r["T"]))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t = self.items[k]
        return _frame_stack(self.records[i], t), int(self.records[i]["label"])


class MiddleFrameDataset(Dataset):
    # Stage 1 sample = the middle frame (T // 2) of a sequence, read from the images; raw intensities as in AppearanceFrameDataset.

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        return _frame_stack(rec, int(rec["T"]) // 2), int(rec["label"])


class CachedFrameDataset(Dataset):
    # Stage 1 sample = the middle frame of a sequence from the local cache written by 01f_cache_stage1_frames.py (uint16 -> float32).

    def __init__(self, split):
        self.path = cfg.STAGE1_CACHE_DIR / f"{split}_frames.npy"
        self.labels = np.load(cfg.STAGE1_CACHE_DIR / f"{split}_labels.npy")
        self.frames = None                                     # opened in the worker process on first use

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if self.frames is None:
            self.frames = np.load(self.path, mmap_mode="r")
        return np.ascontiguousarray(self.frames[i], dtype=np.float32), int(self.labels[i])


class FrameStreamDataset(IterableDataset):
    # Stage 1 frames read sequence by sequence, for a slow disk.

    def __init__(self, records, shuffle, seed=0):
        self.records, self.shuffle, self.seed, self.epoch = records, shuffle, seed, 0
        self.num_frames = sum(int(r["T"]) for r in records)

    def __len__(self):
        return self.num_frames

    def _frames(self, rec):
        sequences = [image_io.read_sequence(rec[f"path_{c}"]) for c in cfg.CHANNELS]
        for t in range(int(rec["T"])):
            stack = image_io.fit_hw(np.stack([image_io.match_z(s[t]) for s in sequences]))
            yield stack, int(rec["label"])

    @staticmethod
    def _as_float(item):
        return np.ascontiguousarray(item[0], dtype=np.float32), item[1]

    def __iter__(self):
        info = get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info else (0, 1)
        order = np.arange(len(self.records))
        if self.shuffle:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.records))
        rng = np.random.default_rng([self.seed, self.epoch, worker])
        buffer = []
        for i in order[worker::n_workers]:
            for item in self._frames(self.records[i]):
                if not self.shuffle:
                    yield self._as_float(item)
                elif len(buffer) < cfg.STREAM_BUFFER:
                    buffer.append(item)
                else:
                    j = int(rng.integers(len(buffer)))
                    buffer[j], item = item, buffer[j]
                    yield self._as_float(item)
        for j in rng.permutation(len(buffer)):                 # end of the epoch: the rest of the buffer, mixed
            yield self._as_float(buffer[j])


def make_stream_loader(dataset, batch_size, drop_last, workers=None):
    # DataLoader for a FrameStreamDataset (every worker streams its own sequences; batches are filled from one worker's stream).
    workers = cfg.NUM_WORKERS if workers is None else workers
    return DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=workers, pin_memory=torch.cuda.is_available())


# ======================================================================================================
# Stage 2: graph expert (all frames, z-projected)
# ======================================================================================================
def project_sequences(sequences):
    # List of (T, Z, H, W) arrays, one per channel -> (C, T, 144, 144) z-projection (maximum over z), padded; integer pixel values are kept.
    if cfg.GRAPH_Z_METHOD != "max":
        raise ValueError("the graph cache and the A549 settings use the maximum projection (GRAPH_Z_METHOD = 'max')")
    return np.stack([image_io.fit_hw(np.max(image_io.match_z(arr), axis=1)) for arr in sequences])


def graph_cache_path(rec):
    # File of the cached projection of an NCM460 sequence, or None (A549 sequences are read from their own fast storage).
    if cfg.GRAPH_CACHE_DIR is None or rec.get("dataset") != "ncm460":
        return None
    return cfg.GRAPH_CACHE_DIR / f"ncm460_{int(rec['row_id'])}.npy"


def cached_projection(rec):
    # (C, T, H, W) projection from the cache, or None when the sequence is not cached.
    path = graph_cache_path(rec)
    return np.load(path) if path is not None and path.exists() else None


def check_cache(records, stage_name):
    # Stop (or warn) when sequences are missing from the cache, so that a long training does not silently read the slow drive.
    missing = sum(1 for r in records if graph_cache_path(r) is not None and not graph_cache_path(r).exists())
    if missing:
        message = f"{missing} of {len(records)} sequences are not in the cache {cfg.GRAPH_CACHE_DIR}: run 01e_cache_graph_inputs.py first ({stage_name})."
        if cfg.GRAPH_CACHE_REQUIRED:
            raise FileNotFoundError(message)
        print("WARNING: " + message + " They are read from the images (slow).")


def normalise_projection(proj, gain=None):
    # (C, T, H, W) projection -> float32 network input: log1p and percentile scaling per channel (after the optional brightness gain).
    proj = calibrate(proj, gain)
    return np.ascontiguousarray(np.stack([image_io.robust_norm(ch) for ch in proj]).astype(np.float32))


def _graph_input(sequences, gain=None):
    # List of (T, Z, H, W) arrays, one per channel -> (C, T, H, W) network input.
    return normalise_projection(project_sequences(sequences), gain)


class GraphSequenceDataset(Dataset):
    # Stage 2 sample = one whole sequence: (x (3, T, H, W), label, T). Read from the projection cache when available.

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        proj = cached_projection(rec)
        x = normalise_projection(proj) if proj is not None else _graph_input([image_io.read_sequence(rec[f"path_{c}"]) for c in cfg.CHANNELS])
        return x, int(rec["label"]), x.shape[1]


# ======================================================================================================
# Stage 3 and evaluation: appearance frame + graph sequence of the same cell
# ======================================================================================================
class FusionDataset(Dataset):
    # Sample = (middle frame as Stage 1 sees it, whole sequence as Stage 2 sees it, label, T). The middle frame is frame T // 2.

    def __init__(self, records, gain=None):
        self.records, self.gain = records, gain

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        t_mid = int(rec["T"]) // 2
        proj = cached_projection(rec)
        if proj is not None:                      # cached graph input: only the middle frame is read from the images
            x_app, x_graph = _frame_stack(rec, t_mid), normalise_projection(proj, self.gain)
        else:
            sequences = [image_io.read_sequence(rec[f"path_{c}"]) for c in cfg.CHANNELS]
            planes = [image_io.match_z(seq[t_mid]) for seq in sequences]
            x_app = np.ascontiguousarray(image_io.fit_hw(np.stack(planes).astype(np.float32)))
            x_graph = _graph_input(sequences, self.gain)
        return calibrate(x_app, self.gain), x_graph, int(rec["label"]), x_graph.shape[1]
