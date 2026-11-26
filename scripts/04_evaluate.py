# Evaluate a Stage 3 checkpoint on one split: appearance only, graph only and the gate variants.
# Example: python scripts/04_evaluate.py --data_root /path/to/npz --fusion_ckpt runs/stage3_gate_*/best_model.pth --split test --save_npz results/test.npz
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdgraph.appearance import AppearanceExpert
from pdgraph.data import CHANNELS, FusionDataset, collate_fusion, index_recordings, read_split_table
from pdgraph.gate import AppearanceGraphGateModel
from pdgraph.temporal_experts import GRAPH_ARCHITECTURES, graph_classifier_class
from pdgraph.utils import parse_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def metric_row(labels, preds):
    return {"top1": 100.0 * float((labels == preds).mean()), "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3")
    parser.add_argument("--csv_path", type=Path, default=Path("data/split_table.csv"))
    parser.add_argument("--data_root", type=str, default="", help="folder with the npz recordings; paths in the split table are relative to it")
    parser.add_argument("--fusion_ckpt", type=Path, required=True, help="best_model.pth of Stage 3")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--graph_channels", nargs="+", default=["mito", "mmp", "nuclear"], choices=CHANNELS)
    parser.add_argument("--min_T", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--app_embed_dim", type=int, default=256)
    parser.add_argument("--graph_embed_dim", type=int, default=128)
    parser.add_argument("--graph_patch_stride", type=int, default=8)
    parser.add_argument("--graph_depth", type=int, default=2)
    parser.add_argument("--graph_num_heads", type=int, default=4)
    parser.add_argument("--graph_dropout", type=float, default=0.15)
    parser.add_argument("--graph_arch", choices=GRAPH_ARCHITECTURES, default="graph", help="architecture of the Stage 2 expert in the checkpoint")
    parser.add_argument("--delta_scale", type=float, default=0.3)
    parser.add_argument("--gate_tau", type=float, default=0.75)
    parser.add_argument("--gate_alpha", type=float, default=10.0)
    parser.add_argument("--fixed_gate", type=float, default=1.0, help="gate value of the constant-gate variant")
    parser.add_argument("--save_npz", type=Path, default=None, help="save logits, gates and embeddings of every recording")
    args = parse_args(parser)

    df, _ = read_split_table(args.csv_path)
    split_df = df[df["split"] == args.split]
    num_classes = int(df["label"].max()) + 1
    items = index_recordings(split_df, args.data_root, CHANNELS, args.min_T, desc=f"Indexing {args.split}")
    loader = DataLoader(FusionDataset(items, args.graph_channels), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fusion)
    print(f"{args.split} samples: {len(items)}")

    state = torch.load(args.fusion_ckpt, map_location="cpu", weights_only=True)
    ckpt_channels = int(state["graph_model.encoder.backbone.net.0.weight"].shape[1])
    if ckpt_channels != len(args.graph_channels):
        raise ValueError(f"the checkpoint expects {ckpt_channels} graph channel(s), got {len(args.graph_channels)}; "
                         "use the channels of training, in the same order")
    gate_out_dim = int(state["gate_head.net.4.weight"].shape[0])
    gate_hidden = int(state["gate_head.net.1.weight"].shape[0])
    if gate_out_dim not in {1, num_classes}:
        raise ValueError(f"unexpected gate output size {gate_out_dim}")
    gate_mode = "classwise" if gate_out_dim == num_classes else "scalar"

    appearance_model = AppearanceExpert(num_classes=num_classes, embed_dim=args.app_embed_dim)
    graph_model = graph_classifier_class(args.graph_arch)(num_classes=num_classes, in_channels=len(args.graph_channels), embed_dim=args.graph_embed_dim,
                                                          patch_stride=args.graph_patch_stride, depth=args.graph_depth,
                                                          num_heads=args.graph_num_heads, dropout=args.graph_dropout)
    model = AppearanceGraphGateModel(appearance_model, graph_model, num_classes=num_classes, graph_embed_dim=args.graph_embed_dim,
                                     delta_scale=args.delta_scale, gate_tau=args.gate_tau, gate_alpha=args.gate_alpha,
                                     gate_mode=gate_mode, gate_hidden=gate_hidden, gate_dropout=0.0)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    names = ["labels", "logits_app", "delta_logits", "gate", "learned_gate", "prior_gate", "graph_embed", "dev_vec", "app_embed",
             "pmax", "margin", "graph_pmax", "graph_margin"]
    out = {k: [] for k in names}
    with torch.no_grad():
        for x_app, x_graph, mask, y in tqdm(loader, desc=f"[eval {args.split}]"):
            (_, logits_app, delta_logits, gate, graph_embed, dev_vec, app_pooled, pmax, margin, learned_gate, prior_gate,
             _, graph_pmax, graph_margin) = model(x_app.to(device), x_graph.to(device), mask.to(device), return_feat=True)
            values = [y, logits_app, delta_logits, gate, learned_gate, prior_gate, graph_embed, dev_vec, app_pooled,
                      pmax.squeeze(1), margin.squeeze(1), graph_pmax.squeeze(1), graph_margin.squeeze(1)]
            for k, v in zip(names, values):
                out[k].append(v.cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    labels, logits_app, delta = out["labels"], out["logits_app"], out["delta_logits"]

    s = args.delta_scale
    logits_by_variant = {
        "appearance": logits_app,
        "graph": delta,
        f"fixed({args.fixed_gate:g})": logits_app + s * args.fixed_gate * delta,
        "prior_only": logits_app + s * out["prior_gate"] * delta,
        "learned_only": logits_app + s * out["learned_gate"] * delta,
        "full_gate": logits_app + s * out["gate"] * delta,
    }
    preds_by_variant = {k: v.argmax(axis=1) for k, v in logits_by_variant.items()}

    print("\n===== Gate variants =====")
    print(f"{'variant':<18} {'top1':>9} {'macro_f1':>11} {'fixed':>8} {'broken':>8} {'net':>8}")
    base = preds_by_variant["appearance"]
    for name, preds in preds_by_variant.items():
        m = metric_row(labels, preds)
        fixed = int(((base != labels) & (preds == labels)).sum())       # wrong for appearance only, right for this variant
        broken = int(((base == labels) & (preds != labels)).sum())
        print(f"{name:<18} {m['top1']:>8.2f}% {m['macro_f1']:>11.4f} {fixed:>8d} {broken:>8d} {fixed - broken:>+8d}")

    gate = out["gate"]
    sample_gate = gate.mean(axis=1) if gate.ndim == 2 else gate
    sample_learned = out["learned_gate"].mean(axis=1) if out["learned_gate"].ndim == 2 else out["learned_gate"]
    app_correct = base == labels
    print("\n===== Gate diagnostics =====")
    print(f"prior beta = {float(F.softplus(model.prior_beta_raw).item()):.4f}")
    print(f"full gate mean/std    = {sample_gate.mean():.4f}/{sample_gate.std():.4f}")
    print(f"learned gate mean/std = {sample_learned.mean():.4f}/{sample_learned.std():.4f}")
    print(f"prior gate mean/std   = {out['prior_gate'].mean():.4f}/{out['prior_gate'].std():.4f}")
    print(f"gate when appearance correct = {sample_gate[app_correct].mean():.4f}")
    print(f"gate when appearance wrong   = {sample_gate[~app_correct].mean():.4f}")
    print(f"correlation of gate with appearance confidence = {np.corrcoef(sample_gate, out['pmax'])[0, 1]:.4f}")
    if gate_mode == "classwise":
        print(f"mean gate by output class = {gate.mean(axis=0).round(4).tolist()}")

    class_names = df.drop_duplicates("label").set_index("label")["class"].to_dict()
    present = sorted(set(labels))
    target_names = [str(class_names.get(v, v)) for v in present]
    for title, preds in (("Appearance only", base), ("Full model", preds_by_variant["full_gate"])):
        print(f"\n===== {title}: per-class report =====")
        print(classification_report(labels, preds, labels=present, target_names=target_names, digits=4, zero_division=0))

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        save = dict(out)
        save["graph_path"] = np.array([it["rel_path"] for it in items], dtype=object)    # relative to the data root
        for name, preds in preds_by_variant.items():
            save["preds_" + name.replace("(", "_").replace(")", "").replace(".", "_")] = preds
        np.savez(args.save_npz, **save)
        print(f"Saved: {args.save_npz}")


if __name__ == "__main__":
    main()
