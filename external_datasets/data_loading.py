# the study index and the datasets (Stage 1, Stage 2, Stage 3 / evaluation) for the public data sets.
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import config as cfg
import image_io
from pdgraph.data import class_weights as class_weights_from_counts
from pdgraph.data import collate_fusion, collate_graph  # noqa: F401  (re-exported for the scripts)
from pdgraph.data import make_loader as make_seeded_loader
from pdgraph.utils import set_seed  # noqa: F401  (re-exported for the scripts)


def load_index(split, min_T=1):
    # Records (list of dicts) of one split of the active protocol. 'test' returns every test part.
    df = pd.read_csv(cfg.INDEX)
    df["row_id"] = df.index
    df["split"] = df[cfg.SPLIT_COLUMN]
    wanted = cfg.SPLIT_ALIASES.get(split, [split])
    df = df[df["split"].isin(wanted) & (df["T"] >= min_T)]
    if cfg.SMOKE:                                              # two sequences per class
        df = df.groupby(["label", "split"], group_keys=False).head(2)
    return df.to_dict("records")


def class_weights(records, power):
    # inverse-frequency class weights from the training records; in a smoke test a class without records gets the smallest count present
    counts = np.bincount([r["label"] for r in records], minlength=cfg.num_classes()).astype(float)
    if np.any(counts == 0):
        if not cfg.SMOKE:
            raise ValueError(f"the training split has no sequences of the classes {np.where(counts == 0)[0].tolist()}")
        counts[counts == 0] = max(counts[counts > 0].min(), 1.0)
    return class_weights_from_counts(counts, power)


def make_loader(dataset, batch_size, shuffle, drop_last, collate_fn=None, seed=0, workers=None):
    return make_seeded_loader(dataset, batch_size, shuffle, drop_last, collate_fn, seed, cfg.NUM_WORKERS if workers is None else workers)


class AnchorFrameDataset(Dataset):
    # Stage 1 sample = the anchor frame of a sequence (LCTP: the 48 h frame; PASTIS: the anchor date; HyperKvasir: the image) and its class.

    def __init__(self, records, augment=False):
        self.records, self.augment = records, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        x = image_io.appearance_input(image_io.load_frame(rec, image_io.anchor_index(rec)))          # (C, 1, H, W)
        if self.augment:                                       # cells have no preferred orientation: random 90-degree rotation and flip
            x = np.rot90(x, np.random.randint(4), axes=(2, 3))
            if np.random.rand() < 0.5:
                x = x[..., ::-1]
            x = np.ascontiguousarray(x)
        return x, int(rec["label"])


class SequenceDataset(Dataset):
    # Stage 2 sample = one whole sequence: (x (C, T, H, W), label, T).

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        x = image_io.graph_input(image_io.load_sequence(rec))
        return x, int(rec["label"]), x.shape[1]


class FusionDataset(Dataset):
    # Stage 3 / evaluation sample = (anchor frame as Stage 1 sees it, whole sequence as Stage 2 sees it, label, T). One read of the sequence serves both.

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        seq = image_io.load_sequence(rec)
        t = min(image_io.anchor_index(rec), seq.shape[1] - 1)
        x_graph = image_io.graph_input(seq)
        return image_io.appearance_input(seq[:, t]), x_graph, int(rec["label"]), x_graph.shape[1]
