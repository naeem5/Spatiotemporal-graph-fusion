# Writes a tiny synthetic data set (random images, same layout as the real recordings) for CPU smoke tests.
# Example: python tests/make_synthetic_data.py --out tmp_synthetic
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Control", "Fission", "Fusion", "Mito/organelle interaction", "mtDNA stress", "MMP uncoupler", "MMP ionophore",
           "ETC complexes", "ROS reduction", "ROS elevation"]
CHANNELS = ["mito", "mmp", "nuclear"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp_synthetic"))
    parser.add_argument("--size", type=int, default=32, help="height and width of the synthetic frames")
    parser.add_argument("--frames", type=int, default=4, help="frames per recording (at least 3)")
    parser.add_argument("--per_class", type=int, default=4, help="recordings per class (split 2 train : 1 val : 1 test when 4)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(args.per_class):
            split = ["train", "train", "val", "test"][k % 4]
            plate, well, field, name = f"plate{k % 2 + 1}", f"W{label:02d}{k}", "Field1", f"{label * 10 + k}.npz"
            for ch in CHANNELS:
                rel = Path(plate) / ch / f"{well}_{field}" / name
                path = args.out / "npz" / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                img = (rng.gamma(2.0, 200.0, size=(args.frames, 27, args.size, args.size)) + 100 * label).astype(np.uint16)
                np.savez(path, img=img)
                rows.append(dict(channel=ch, split=split, plate=plate, well=well, field=field, filename=name, path=rel.as_posix(),
                                 compound=f"compound{label}", label=label, **{"class": cls}))
    pd.DataFrame(rows).to_csv(args.out / "split_table.csv", index=False)
    print(f"wrote {len(rows) // 3} recordings of {len(CLASSES)} classes to {args.out}")


if __name__ == "__main__":
    main()
