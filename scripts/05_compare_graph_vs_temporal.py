# Does the patch graph add to the fusion beyond access to the sequence? Compares the gated fusion built on the patch graph with the fusion
# built on a temporal-only control expert (same Stage 1, same gate training), from the arrays written by 04_evaluate.py.
# Example: python scripts/05_compare_graph_vs_temporal.py --graph_results results/graph_seed{1234,42,2025}.npz \
#              --control temporal results/temporal_seed1234.npz results/temporal_seed42.npz results/temporal_seed2025.npz
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pdgraph.metrics import macro_f1_from_confusion  # noqa: E402

NUM_CLASSES = 10
DISPLAY_NAME = {"graph": "patch graph", "temporal": "temporal (one vector per frame)", "temporal_nodes": "temporal (nodes, no spatial edges)"}


def well_id(recording_path):
    # "a549-p1/mito/WellB02_Channel..._Field1/008.npz" -> "a549-p1|WellB02"
    parts = recording_path.split("/")
    return parts[0] + "|" + parts[2].split("_")[0]


def confusion_per_well(labels, predictions, well_index, n_wells):
    confusion = np.zeros((n_wells, NUM_CLASSES, NUM_CLASSES))
    np.add.at(confusion, (well_index, labels.astype(int), predictions.astype(int)), 1)
    return confusion


def top1_and_macro_f1_of_confusion(confusion):
    return 100.0 * np.trace(confusion) / confusion.sum(), macro_f1_from_confusion(confusion)


def load_gate_runs(files):
    runs = [dict(np.load(f, allow_pickle=True)) for f in files]
    for run in runs[1:]:        # the gates share the frozen experts, so everything except the gate must agree
        for key in ("labels", "logits_app", "delta_logits", "preds_appearance", "preds_graph"):
            if not np.allclose(runs[0][key].astype(float), run[key].astype(float), atol=1e-4):
                raise RuntimeError(f"the gate runs {files} disagree in {key}")
    return runs


def main():
    parser = argparse.ArgumentParser(description="Compare the patch graph with temporal-only control experts")
    parser.add_argument("--graph_results", type=Path, nargs=3, required=True, help="test arrays of the three gates built on the patch graph")
    parser.add_argument("--control", nargs=4, action="append", metavar=("NAME", "SEED1", "SEED2", "SEED3"), required=True,
                        help="NAME (temporal or temporal_nodes) and the test arrays of its three gates; repeat for several controls")
    parser.add_argument("--out_dir", type=Path, default=Path("results"))
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    arms = {"graph": load_gate_runs(args.graph_results)}
    for name, *files in args.control:
        arms[name] = load_gate_runs([Path(f) for f in files])
    controls = [name for name in arms if name != "graph"]

    reference = arms["graph"][0]
    labels = reference["labels"].astype(int)
    for name in controls:
        run = arms[name][0]
        same_cells = np.array_equal(run["labels"], reference["labels"]) and np.array_equal(run["graph_path"].astype(str), reference["graph_path"].astype(str))
        if not same_cells:
            raise RuntimeError(f"{name}: different test recordings than the patch-graph arm")
        if not np.allclose(run["logits_app"], reference["logits_app"], atol=1e-4):
            raise RuntimeError(f"{name}: the appearance expert differs from the patch-graph arm, so the arms are not comparable")

    wells, well_index = np.unique([well_id(p) for p in reference["graph_path"].astype(str)], return_inverse=True)
    n_wells = len(wells)
    confusion = {"appearance": confusion_per_well(labels, reference["preds_appearance"], well_index, n_wells)}
    for arm, runs in arms.items():
        confusion[(arm, "expert")] = confusion_per_well(labels, runs[0]["preds_graph"], well_index, n_wells)
        for i, run in enumerate(runs):
            confusion[(arm, "fused", i)] = confusion_per_well(labels, run["preds_full_gate"], well_index, n_wells)

    def estimate(well_weights):
        total = {key: np.tensordot(well_weights, value, 1) for key, value in confusion.items()}
        appearance = np.array(top1_and_macro_f1_of_confusion(total["appearance"]))
        expert = {arm: np.array(top1_and_macro_f1_of_confusion(total[(arm, "expert")])) for arm in arms}
        fused = {arm: np.mean([top1_and_macro_f1_of_confusion(total[(arm, "fused", i)]) for i in range(3)], axis=0) for arm in arms}
        result = {}
        for j, metric in enumerate(("top-1", "macro-F1")):
            for arm in arms:
                result[f"{metric}: gain of the gated fusion over the appearance expert, {DISPLAY_NAME[arm]}"] = fused[arm][j] - appearance[j]
            for arm in controls:
                result[f"{metric}: DIFFERENCE of the two gains (patch graph minus {DISPLAY_NAME[arm]})"] = fused["graph"][j] - fused[arm][j]
                result[f"{metric}: Stage 2 expert alone, patch graph minus {DISPLAY_NAME[arm]}"] = expert["graph"][j] - expert[arm][j]
            for arm in arms:
                result[f"{metric}: gain of the gated fusion over its own Stage 2 expert, {DISPLAY_NAME[arm]}"] = fused[arm][j] - expert[arm][j]
        return result, appearance, expert, fused

    point, appearance, expert, fused = estimate(np.ones(n_wells))
    rng = np.random.default_rng(args.seed)
    draws = {key: np.empty(args.n_boot) for key in point}
    for b in range(args.n_boot):
        for key, value in estimate(np.bincount(rng.integers(0, n_wells, n_wells), minlength=n_wells).astype(float))[0].items():
            draws[key][b] = value

    lines = ["# A549 test set: patch graph versus temporal-only control expert(s)", "",
             f"Test recordings: {len(labels)}; wells (bootstrap clusters): {n_wells}; fusion = mean over three gates; the appearance expert is identical in all arms.", "",
             "```", f"{'':<58}{'top-1 (%)':>10}{'macro-F1':>10}", f"{'appearance expert (Stage 1)':<58}{appearance[0]:>10.2f}{appearance[1]:>10.3f}"]
    for arm in arms:
        lines.append(f"{'Stage 2 expert alone, ' + DISPLAY_NAME[arm]:<58}{expert[arm][0]:>10.2f}{expert[arm][1]:>10.3f}")
        lines.append(f"{'gated fusion, ' + DISPLAY_NAME[arm]:<58}{fused[arm][0]:>10.2f}{fused[arm][1]:>10.3f}")
    lines += ["```", "", "## Paired differences, 95% percentile intervals from resampling whole wells", "", "```",
              f"{'quantity':<110}{'estimate':>10}{'ci_low':>9}{'ci_high':>9}"]
    for key in point:
        low, high = np.percentile(draws[key], [2.5, 97.5])
        lines.append(f"{key:<110}{point[key]:>10.4f}{low:>9.4f}{high:>9.4f}")
    lines += ["```", "", "An interval of the difference above 0 means that the patch graph contributes beyond temporal access; a control gain of similar size with a "
              "difference near 0 means that the gain comes from the sequence input."]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "graph_vs_temporal.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
