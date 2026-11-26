from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CHANNELS = ["mito", "mmp", "nuclear"]
RECORDING_KEYS = ["plate", "well", "field", "filename"]     # columns of the split table that identify one recording


def load_array(path):
    d = np.load(path, mmap_mode="r")
    return d["img"] if "img" in d else d[d.files[0]]       # (T, Z, H, W)


def pad_hw(arr, size=144):
    # zero padding of the last two dimensions at the bottom and right
    pad_h, pad_w = max(0, size - arr.shape[-2]), max(0, size - arr.shape[-1])
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)], mode="constant", constant_values=0)


def fit_hw(arr, size=144):
    # last two axes -> size x size: zero padding at the bottom and right when smaller, centre crop when larger
    height, width = arr.shape[-2:]
    if height > size:
        top = (height - size) // 2
        arr = arr[..., top:top + size, :]
    if width > size:
        left = (width - size) // 2
        arr = arr[..., :, left:left + size]
    height, width = arr.shape[-2:]
    if height < size or width < size:
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, size - height), (0, size - width)], mode="constant", constant_values=0)
    return arr


def make_loader(dataset, batch_size, shuffle, drop_last, collate_fn=None, seed=0, workers=0):
    # DataLoader with pinned memory on GPU machines, seeded workers and reproducible shuffling
    from pdgraph.utils import set_seed
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=workers, pin_memory=torch.cuda.is_available())
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    if workers > 0:
        kwargs["worker_init_fn"] = set_seed(seed)
    return DataLoader(dataset, **kwargs)


def robust_norm(arr, q_low=0.01, q_high=0.99, eps=1e-6):
    # log transform, then clip to the 1st-99th percentile of the whole array and rescale to [0, 1]
    x = np.log1p(arr.astype(np.float32))
    flat = x.reshape(-1)
    lo, hi = np.quantile(flat, q_low), np.quantile(flat, q_high)
    if hi - lo < eps:
        return np.zeros_like(x)
    return (np.clip(x, lo, hi) - lo) / (hi - lo + eps)


def index_recordings(df, data_root="", channels=CHANNELS, min_T=1, desc="Indexing recordings"):
    # one entry per recording that has all requested channels, 4-D arrays and the same number of frames in every channel
    items = []
    grouped = df.groupby(RECORDING_KEYS)
    for _, g in tqdm(grouped, total=grouped.ngroups, desc=desc):
        g = g.set_index("channel")
        if any(c not in g.index for c in channels):
            continue
        paths = {c: str(Path(data_root) / g.loc[c, "path"]) for c in channels}
        frames, ok = [], True
        for c in channels:
            arr = load_array(paths[c])
            if arr.ndim != 4:
                ok = False
                break
            frames.append(arr.shape[0])
        if not ok or len(set(frames)) != 1 or frames[0] < min_T:
            continue
        items.append({"paths": paths, "rel_path": g.loc[channels[0], "path"], "label": int(g.iloc[0]["label"]), "T": frames[0]})
    return items


class AppearanceFrameDataset(Dataset):
    # every frame of every recording is a sample: raw (3, Z, H, W) z-stack
    def __init__(self, items):
        self.items = items
        self.index = [(i, t) for i, it in enumerate(items) for t in range(it["T"])]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        item = self.items[i]
        sample = np.stack([load_array(item["paths"][c])[t].copy() for c in CHANNELS], axis=0).astype(np.float32)
        return np.ascontiguousarray(pad_hw(sample)), item["label"]


def graph_input(item, channels):
    # (C, T, H, W): maximum projection over z, padding, and an independent normalisation of every channel
    chans = []
    for c in channels:
        proj = np.max(np.asarray(load_array(item["paths"][c])), axis=1)
        chans.append(robust_norm(pad_hw(proj)))
    return np.ascontiguousarray(np.stack(chans, axis=0).astype(np.float32))


class GraphSequenceDataset(Dataset):
    def __init__(self, items, channels):
        self.items, self.channels = items, list(channels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        sample = graph_input(item, self.channels)
        return sample, item["label"], sample.shape[1]


class FusionDataset(Dataset):
    # appearance input: the middle frame as raw z-stack; graph input: the whole recording
    def __init__(self, items, graph_channels):
        self.items, self.graph_channels = items, list(graph_channels)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        t_mid = item["T"] // 2
        x_app = np.stack([load_array(item["paths"][c])[t_mid].copy() for c in CHANNELS], axis=0).astype(np.float32)
        x_app = np.ascontiguousarray(pad_hw(x_app))
        return x_app, graph_input(item, self.graph_channels), item["label"], item["T"]


def _pad_sequences(xs):
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    C, _, H, W = xs[0].shape
    x_pad = torch.zeros((len(xs), C, int(lengths.max()), H, W), dtype=torch.float32)
    mask = torch.zeros((len(xs), int(lengths.max())), dtype=torch.bool)
    for i, x in enumerate(xs):
        x_pad[i, :, : x.shape[1]] = torch.from_numpy(x)
        mask[i, : x.shape[1]] = True
    return x_pad, mask, lengths


def collate_graph(batch):
    xs, ys, _ = zip(*batch)
    x_pad, mask, lengths = _pad_sequences(xs)
    return x_pad, mask, lengths, torch.tensor(ys, dtype=torch.long)


def collate_fusion(batch):
    x_apps, x_graphs, ys, _ = zip(*batch)
    x_graph, mask, _ = _pad_sequences(x_graphs)
    return torch.from_numpy(np.stack(x_apps, axis=0)), x_graph, mask, torch.tensor(ys, dtype=torch.long)


def class_weights(counts, power):
    counts = np.asarray(counts)
    return torch.tensor(((counts.sum() / len(counts)) / counts) ** power, dtype=torch.float32)


def read_split_table(path, split_names=("train", "val", "test")):
    df = pd.read_csv(path)
    return df, [df[df["split"] == s] for s in split_names]
