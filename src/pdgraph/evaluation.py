import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pdgraph.metrics import top1_and_macro_f1

# outputs of the two frozen experts that the gate needs; they never change during Stage 3, so they are computed once and stored
FEATURE_KEYS = ["logits_app", "app_pooled", "graph_logits", "graph_embed", "dev_vec", "z_graph"]
# per-recording outputs of a complete Stage 3 model
OUTPUT_KEYS = ["labels", "logits_app", "delta_logits", "gate", "learned_gate", "prior_gate", "graph_embed", "dev_vec", "app_embed",
               "pmax", "margin", "graph_pmax", "graph_margin"]


@torch.no_grad()
def expert_features(model, loader, device):
    # run the frozen experts of a Stage 3 model over a loader of (x_app, x_graph, mask, label); arrays of FEATURE_KEYS and the labels
    store = {k: [] for k in FEATURE_KEYS}
    labels = []
    model.eval()
    for x_app, x_graph, mask, y in tqdm(loader, desc="reading images, running the experts"):
        features = model.expert_features(x_app.to(device), x_graph.to(device), mask.to(device))
        for k in FEATURE_KEYS:
            store[k].append(features[k].float().cpu().numpy())
        labels.append(y.numpy())
    result = {k: np.concatenate(v) for k, v in store.items()}
    result["labels"] = np.concatenate(labels)
    return result


def load_or_compute_features(folder, split, rows, meta, compute):
    # expert outputs of a split, stored in folder/<split>.npz and reused when the recordings (row ids) and the experts (meta) are the same
    folder.mkdir(parents=True, exist_ok=True)
    path, meta_path = folder / f"{split}.npz", folder / f"{split}_meta.json"
    if path.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == meta:
        stored = dict(np.load(path, allow_pickle=True))
        if np.array_equal(stored["row_id"], rows):
            print(f"[{split}] expert outputs loaded from {path}")
            return stored
    features = compute()
    np.savez(path, **features)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"[{split}] expert outputs saved to {path}")
    return features


@torch.no_grad()
def fused_outputs(model, features, device, batch_size=1024):
    # gate and fusion of a Stage 3 model on stored expert outputs; arrays of OUTPUT_KEYS
    store = {k: [] for k in OUTPUT_KEYS}
    model.eval()
    for i in range(0, len(features["labels"]), batch_size):
        batch = {k: torch.from_numpy(np.asarray(features[k][i:i + batch_size])).to(device) for k in FEATURE_KEYS}
        (_, logits_app, delta_logits, gate, graph_embed, dev_vec, app_pooled, pmax, margin, learned_gate, prior_gate, _, graph_pmax,
         graph_margin) = model.fuse(batch, return_feat=True)
        values = [features["labels"][i:i + batch_size], logits_app, delta_logits, gate, learned_gate, prior_gate, graph_embed, dev_vec, app_pooled,
                  pmax.squeeze(1), margin.squeeze(1), graph_pmax.squeeze(1), graph_margin.squeeze(1)]
        for k, v in zip(OUTPUT_KEYS, values):
            store[k].append(v if isinstance(v, np.ndarray) else v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in store.items()}


def average_gates(outputs):
    # mean of the gates of several seeds; the experts are frozen and identical, so everything else is taken from the first seed
    merged = dict(outputs[0])
    for k in ("gate", "learned_gate", "prior_gate"):
        merged[k] = np.mean([o[k] for o in outputs], axis=0)
    return merged


def variant_table(labels, predictions, reference="appearance", with_count=False):
    # one row per variant: top-1 (%), macro-F1 and the recordings it corrects / breaks compared with the reference variant
    labels = np.asarray(labels)
    base = predictions[reference]
    rows = []
    for name, p in predictions.items():
        top1, macro_f1 = top1_and_macro_f1(labels, p)
        row = dict(variant=name)
        if with_count:
            row["n"] = len(labels)
        row.update(top1=top1, macro_f1=macro_f1, fixed=int(((base != labels) & (p == labels)).sum()), broken=int(((base == labels) & (p != labels)).sum()))
        rows.append(row)
    return pd.DataFrame(rows)
