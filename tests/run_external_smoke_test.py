# Smoke test of external_datasets on random HyperKvasir-like images (CPU, one minute): trains the three stages for one epoch and evaluates.
# Example: python tests/run_external_smoke_test.py --work tmp_external_smoke
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "external_datasets"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=ROOT / "tmp_external_smoke")
    work = parser.parse_args().work.resolve()
    csv_dir, out_dir, cache = work / "csv", work / "out", work / "cache"
    (cache / "hyperkvasir").mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    classes, per_class = ["normal-cecum", "normal-pylorus", "polyps", "ulcerative-colitis"], 10
    rows = []
    for label, name in enumerate(classes):
        for k in range(per_class):
            rows.append(dict(file=f"{name}_{k}", cls=name, label=label, cache_row=len(rows), well=f"{name}_{k}", T=1, anchor=0,
                             **{f"split_official5_f{f}": ["test" if k % 5 == f else "val" if k % 5 == (f + 1) % 5 else "train"][0] for f in range(5)}))
    pd.DataFrame(rows).to_csv(csv_dir / "hyperkvasir_index.csv", index=False)
    (csv_dir / "hyperkvasir_classes.json").write_text(json.dumps(dict(names=classes, control_label=0, control_labels=[0, 1])), encoding="utf-8")
    np.save(cache / "hyperkvasir" / "images_144.npy", rng.integers(0, 255, size=(len(rows), 3, 144, 144), dtype=np.uint8))

    env = dict(os.environ, PUBLIC_DATASET="hyperkvasir", PUBLIC_SMOKE="1", PUBLIC_CSV_DIR=str(csv_dir), PUBLIC_OUT_DIR=str(out_dir),
               PUBLIC_TILE_CACHE=str(cache), PUBLIC_RETRAIN="1")
    for script in ("04_train_stage1_appearance.py", "05_train_stage2_graph.py", "06_train_stage3_gate.py", "07_evaluate.py"):
        print("$", script)
        subprocess.run([sys.executable, script], check=True, cwd=EXT, env=env)
    # temporal-only control experts reuse the Stage 1 of the graph run
    for architecture in ("temporal", "temporal_nodes"):
        for script in ("05_train_stage2_graph.py", "06_train_stage3_gate.py", "07_evaluate.py"):
            print("$", architecture, script)
            subprocess.run([sys.executable, script], check=True, cwd=EXT, env=dict(env, PUBLIC_STAGE2_ARCH=architecture))
        assert (out_dir / "reports" / f"hyperkvasir_official5_f0_{architecture}_smoke" / "results.md").exists(), f"no results for {architecture}"
    print("external smoke test finished")


if __name__ == "__main__":
    main()
