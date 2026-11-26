# index of the PASTIS patches (Sentinel-2 satellite image time series of agricultural land): patch label, anchor acquisitions, official folds.
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config as cfg

PASTIS_MIN_PATCHES = 20
PASTIS_CLASSES = {1: "meadow", 2: "soft_winter_wheat", 3: "corn", 4: "winter_barley", 5: "winter_rapeseed", 6: "spring_barley", 7: "sunflower", 8: "grapevine", 9: "beet",
                  10: "winter_triticale", 11: "winter_durum_wheat", 12: "fruits_vegetables_flowers", 13: "potatoes", 14: "leguminous_fodder", 15: "soybeans", 16: "orchard",
                  17: "mixed_cereal", 18: "sorghum"}                       # background = 0, void = 19 (names as in the data set documentation and in torchgeo.datasets.PASTIS)
REFERENCE_CLASS_ID = 1
N_BANDS = 10


def read_metadata(root):
    meta = json.loads((root / "metadata.geojson").read_text(encoding="utf-8"))
    rows = []
    for feature in meta["features"]:
        p = feature["properties"]
        missing = [k for k in ("ID_PATCH", "Fold", "dates-S2") if k not in p]
        if missing:
            raise KeyError(f"metadata.geojson has no {missing}; the properties of the first patch are {sorted(p)}")
        dates = p["dates-S2"]
        dates = json.loads(dates) if isinstance(dates, str) else dates
        keys = sorted(dates, key=int) if all(str(k).lstrip("-").isdigit() for k in dates) else list(dates)
        rows.append(dict(patch_id=int(p["ID_PATCH"]), fold=int(p["Fold"]), dates=[datetime.strptime(str(dates[k]), "%Y%m%d") for k in keys]))
    return pd.DataFrame(rows).sort_values("patch_id").reset_index(drop=True)


def closest(dates, target):
    # Index of the acquisition closest to the target date and its distance in days.
    days = np.array([abs((d - target).days) for d in dates])
    return int(days.argmin()), int(days.min())


def main():
    if not cfg.IS_PASTIS:
        sys.exit("set PUBLIC_DATASET=pastis")
    root, allow_missing = cfg.PASTIS_ROOT, os.environ.get("PUBLIC_PASTIS_ALLOW_MISSING", "0") == "1"
    meta = read_metadata(root)
    print(f"{len(meta)} patches in metadata.geojson; folds: {meta['fold'].value_counts().sort_index().to_dict()}")

    rows, skipped = [], 0
    for r in meta.itertuples():
        data_file = root / "DATA_S2" / f"S2_{r.patch_id}.npy"
        if not data_file.exists():
            if allow_missing:
                skipped += 1
                continue
            raise FileNotFoundError(f"{data_file} is missing (PUBLIC_PASTIS_ALLOW_MISSING=1 only for code tests on a subset)")
        shape = np.load(data_file, mmap_mode="r").shape
        if len(shape) != 4 or shape[0] != len(r.dates) or shape[1] != N_BANDS:
            raise ValueError(f"{data_file.name}: shape {shape} does not match {len(r.dates)} acquisitions x {N_BANDS} bands")
        if any(b < a for a, b in zip(r.dates, r.dates[1:])):
            raise ValueError(f"patch {r.patch_id}: the acquisition dates are not in chronological order")
        target = np.load(root / "ANNOTATIONS" / f"TARGET_{r.patch_id}.npy")
        counts = np.bincount(target[0].astype(np.int64).ravel(), minlength=20)
        crop = counts[1:19]
        n_crop = int(crop.sum())
        row = dict(patch_id=r.patch_id, fold=r.fold, T=len(r.dates), n_crop_pixels=n_crop, share_crop=n_crop / counts.sum(),
                   class_id=int(crop.argmax()) + 1 if n_crop else 0, purity=float(crop.max() / n_crop) if n_crop else 0.0,
                   first_date=r.dates[0].date().isoformat(), last_date=r.dates[-1].date().isoformat())
        for tag, day in cfg.PASTIS_ANCHOR_DATES.items():
            i, off = closest(r.dates, datetime.strptime(day, "%Y-%m-%d"))
            row[f"anchor_{tag}"], row[f"anchor_{tag}_offset_days"] = i, off
        rows.append(row)
    df = pd.DataFrame(rows)
    if skipped:
        print(f"WARNING: {skipped} patches skipped because their DATA_S2 file is missing (code test only)")

    no_crop = int((df["class_id"] == 0).sum())
    df = df[df["class_id"] > 0]
    counts = df["class_id"].value_counts()
    rare = sorted(counts[counts < PASTIS_MIN_PATCHES].index.tolist())
    removed = int(df["class_id"].isin(rare).sum())
    df = df[~df["class_id"].isin(rare)].copy()
    kept = sorted(df["class_id"].unique())
    if REFERENCE_CLASS_ID not in kept:
        raise ValueError(f"the reference class {REFERENCE_CLASS_ID} (meadow) is not among the kept classes {kept}")
    label_of = {c: i for i, c in enumerate(kept)}
    df["label"] = df["class_id"].map(label_of)
    df["class"] = df["class_id"].map(PASTIS_CLASSES)
    df["well"] = "patch" + df["patch_id"].astype(str)                     # the group of the bootstrap: one patch
    for k in range(5):
        test, val = k + 1, (k + 1) % 5 + 1
        df[f"split_official5_f{k}"] = np.where(df["fold"] == test, "test", np.where(df["fold"] == val, "val", "train"))
    df = df.reset_index(drop=True)
    for k in range(5):
        missing = set(range(len(kept))) - set(df.loc[df[f"split_official5_f{k}"] == "train", "label"])
        if missing and not skipped:
            raise ValueError(f"run {k}: no training patch for the classes {sorted(kept[m] for m in missing)}")

    cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["patch_id", "fold", "class_id", "class", "label", "purity", "n_crop_pixels", "share_crop", "T", "first_date", "last_date", "well"]
    cols += [c for c in df.columns if c.startswith("anchor_")] + [f"split_official5_f{k}" for k in range(5)]
    df[cols].to_csv(cfg.INDEX, index=False)
    names = [PASTIS_CLASSES[c] for c in kept]
    cfg.CLASSES_JSON.write_text(json.dumps(dict(names=names, class_ids=[int(c) for c in kept], counts=[int((df["class_id"] == c).sum()) for c in kept],
                                                control_label=label_of[REFERENCE_CLASS_ID], control_labels=[label_of[REFERENCE_CLASS_ID]],
                                                removed_class_ids=[int(c) for c in rare], patches_without_crop=no_crop), indent=1), encoding="utf-8")
    shutil.copy(root / "NORM_S2_patch.json", cfg.PASTIS_NORM_FILE)

    cfg.REPORT_BASE.mkdir(parents=True, exist_ok=True)
    per_class = df.groupby(["class_id", "class"]).agg(patches=("patch_id", "size"), median_purity=("purity", "median"), median_T=("T", "median")).reset_index()
    per_fold = df.pivot_table(index="class", columns="fold", values="patch_id", aggfunc="count", fill_value=0)
    lines = ["# PASTIS: what the index contains (written by 03_build_index_pastis.py before any model was trained)", "",
             f"Patches in metadata.geojson: {len(meta)}; without any crop pixel (dropped): {no_crop}; in classes with fewer than {PASTIS_MIN_PATCHES} patches (dropped): {removed}"
             f" (class ids {rare}); patches used: {len(df)}; classes: {len(kept)}; reference class: meadow (label {label_of[REFERENCE_CLASS_ID]}).", "",
             "## Patches per class", "", "```", per_class.round(3).to_string(index=False), "```", "", "## Patches per class and official fold", "", "```", per_fold.to_string(), "```", "",
             f"Purity (share of the crop pixels that belong to the dominant crop): quartiles {df['purity'].quantile([.25, .5, .75]).round(3).tolist()}; patches with purity < 0.5: "
             f"{int((df['purity'] < 0.5).sum())}. Share of the patch that is crop: median {df['share_crop'].median():.3f}.", "",
             f"Acquisitions per patch: min {df['T'].min()}, median {int(df['T'].median())}, max {df['T'].max()}. First acquisition {df['first_date'].min()}, last {df['last_date'].max()}.", ""]
    for tag, day in cfg.PASTIS_ANCHOR_DATES.items():
        off = df[f"anchor_{tag}_offset_days"]
        lines.append(f"Anchor '{tag}' = acquisition closest to {day}: distance in days min {off.min()}, median {int(off.median())}, max {off.max()}; patches more than 10 days away: {int((off > 10).sum())}.")
    (cfg.REPORT_BASE / "pastis_data_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {cfg.INDEX}, {cfg.CLASSES_JSON}, {cfg.PASTIS_NORM_FILE}, {cfg.REPORT_BASE / 'pastis_data_report.md'}")


if __name__ == "__main__":
    main()
