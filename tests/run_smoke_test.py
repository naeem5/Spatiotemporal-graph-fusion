# End-to-end check on random synthetic data (CPU, a few minutes): trains the three stages for 2 epochs, evaluates, trains the control expert and compares.
# Example: python tests/run_smoke_test.py --work tmp_smoke
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]


def run(*args):
    print("$", " ".join(str(a) for a in args))
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=ROOT)


def newest(folder, pattern):
    return sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=ROOT / "tmp_smoke")
    args = parser.parse_args()
    work = args.work.resolve()
    runs = work / "runs"

    run("tests/make_synthetic_data.py", "--out", work / "data")
    common = ["--csv_path", work / "data" / "split_table.csv", "--data_root", work / "data" / "npz", "--out_dir", runs, "--num_workers", "0", "--epochs", "2"]
    run("scripts/01_train_stage1_appearance.py", "--config", "configs/a549_stage1_appearance.yaml", *common, "--batch_size", "8")
    run("scripts/02_train_stage2_graph.py", "--config", "configs/a549_stage2_graph.yaml", *common, "--batch_size", "4")
    stage1 = newest(runs, "stage1_appearance_*") / "best_model.pth"
    stage2 = newest(runs, "stage2_graph_*") / "best_model.pth"
    run("scripts/03_train_stage3_gate.py", "--config", "configs/a549_stage3_gate_seed1234.yaml", *common, "--batch_size", "4",
        "--app_ckpt", stage1, "--graph_ckpt", stage2)
    stage3 = newest(runs, "stage3_gate_*") / "best_model.pth"
    eval_npz = work / "stage3_seed1234_test.npz"
    run("scripts/04_evaluate.py", "--csv_path", work / "data" / "split_table.csv", "--data_root", work / "data" / "npz", "--fusion_ckpt", stage3,
        "--split", "test", "--batch_size", "4", "--num_workers", "0", "--save_npz", eval_npz)

    # temporal-only control expert: Stage 2, Stage 3 and evaluation with --graph_arch temporal, then the comparison with the patch graph
    run("scripts/02_train_stage2_graph.py", "--config", "configs/a549_stage2_graph.yaml", *common, "--batch_size", "4", "--graph_arch", "temporal")
    control_stage2 = newest(runs, "stage2_graph_*") / "best_model.pth"
    run("scripts/03_train_stage3_gate.py", "--config", "configs/a549_stage3_gate_seed1234.yaml", *common, "--batch_size", "4",
        "--app_ckpt", stage1, "--graph_ckpt", control_stage2, "--graph_arch", "temporal")
    control_stage3 = newest(runs, "stage3_gate_*") / "best_model.pth"
    control_npz = work / "temporal_seed1234_test.npz"
    run("scripts/04_evaluate.py", "--csv_path", work / "data" / "split_table.csv", "--data_root", work / "data" / "npz", "--fusion_ckpt", control_stage3,
        "--split", "test", "--batch_size", "4", "--num_workers", "0", "--graph_arch", "temporal", "--save_npz", control_npz)
    run("scripts/05_compare_graph_vs_temporal.py", "--graph_results", eval_npz, eval_npz, eval_npz, "--control", "temporal", control_npz, control_npz, control_npz,
        "--out_dir", work / "results", "--n_boot", "50")

    print("smoke test finished")


if __name__ == "__main__":
    main()
