# index and image cache of the HyperKvasir labeled images (10,662 endoscopy images, 23 classes).
import json
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config as cfg

ZIP_NAME = "hyper-kvasir-labeled-images.zip"


def list_entries():
    # {file name: reader} for the images of the zip (or of the unzipped folder).
    zip_path = cfg.HK_ROOT / ZIP_NAME
    if zip_path.exists():
        zf = zipfile.ZipFile(zip_path)
        names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".jpg")}
        return names, (lambda n: zf.read(n)), f"zip {zip_path}"
    folder = cfg.HK_ROOT / "labeled-images"
    if folder.is_dir():
        names = {p.name: p for p in folder.rglob("*.jpg")}
        return names, (lambda p: Path(p).read_bytes()), f"folder {folder}"
    raise FileNotFoundError(f"neither {zip_path} nor {folder} exists: set PUBLIC_HK_ROOT to the folder with {ZIP_NAME}")


def decode(raw):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode an image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg.TARGET_HW, cfg.TARGET_HW), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def main():
    if not cfg.IS_HK:
        sys.exit("set PUBLIC_DATASET=hyperkvasir")
    split = pd.read_csv(cfg.HK_SPLIT_FILE, sep=";")
    split = split.rename(columns={"file-name": "file", "class-name": "class", "split-index": "fold"})
    names = sorted(split["class"].unique())
    label_of = {n: i for i, n in enumerate(names)}
    missing = [c for c in cfg.HK_NORMAL_CLASSES if c not in label_of]
    if missing:
        raise ValueError(f"reference classes not in the data: {missing}")
    split["label"] = split["class"].map(label_of)
    split["well"] = split["file"]
    split["T"], split["anchor"] = 1, 0
    split = split.reset_index(drop=True)
    split["cache_row"] = np.arange(len(split))
    for k in range(5):
        split[f"split_official5_f{k}"] = np.where(split["fold"] == k, "test", np.where(split["fold"] == (k + 1) % 5, "val", "train"))

    entries, read, source = list_entries()
    absent = [f for f in split["file"] if f not in entries]
    if absent:
        raise FileNotFoundError(f"{len(absent)} files of the split are not in the {source}, e.g. {absent[:3]}")
    print(f"source: {source}; {len(split)} images, {len(names)} classes")

    # class names must agree with the folders of the zip / folder (the split file and the data were published separately)
    folder_class = {f: (entries[f].split("/")[-2] if isinstance(entries[f], str) else entries[f].parent.name) for f in split["file"]}
    # The split file spells 8 class names differently from the folders (hemorroids/hemorrhoids, normal-cecum/cecum, normal-pylorus/pylorus, normal-z-line/z-line,
    # oesophagitis-*/esophagitis-*, polyp/polyps, short-segment-barretts/barretts-short-segment); the other 15 are identical. What is required is a one-to-one mapping.
    pairs = pd.DataFrame({"class": split["class"], "folder": split["file"].map(folder_class)}).drop_duplicates()
    if pairs["class"].duplicated().any() or pairs["folder"].duplicated().any():
        raise ValueError("a class of the split file is spread over several folders (or the reverse):\n" + pairs.sort_values("class").to_string())
    print("class name in the split file -> folder of the zip, where they differ: " +
          "; ".join(f"{r['class']} -> {r['folder']}" for _, r in pairs.iterrows() if r["class"] != r["folder"]))

    cfg.HK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(cfg.HK_CACHE, mode="w+", dtype=np.uint8, shape=(len(split), 3, cfg.TARGET_HW, cfg.TARGET_HW))
    step, files = 256, split["file"].tolist()
    with ThreadPoolExecutor(max_workers=cfg.BUILD_WORKERS) as pool:
        for start in range(0, len(files), step):
            raws = [read(entries[f]) for f in files[start:start + step]]           # sequential read (one zip handle), parallel decode
            for i, arr in enumerate(pool.map(decode, raws)):
                cache[start + i] = arr
            if (start // step) % 8 == 0:
                print(f"  {start + len(raws)} / {len(files)}")
    cache.flush()
    del cache

    cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["file", "class", "label", "cache_row", "well", "T", "anchor"] + [f"split_official5_f{k}" for k in range(5)]
    split[keep].to_csv(cfg.INDEX, index=False)
    counts = split["class"].value_counts().reindex(names).tolist()
    cfg.CLASSES_JSON.write_text(json.dumps(dict(names=names, counts=counts, control_label=label_of[cfg.HK_NORMAL_CLASSES[0]],
                                                control_labels=[label_of[c] for c in cfg.HK_NORMAL_CLASSES]), indent=1), encoding="utf-8")
    print(f"written: {cfg.INDEX}, {cfg.CLASSES_JSON}, {cfg.HK_CACHE}")
    for k in range(5):
        table = split.pivot_table(index="class", columns=f"split_official5_f{k}", values="file", aggfunc="count", fill_value=0)
        print(f"fold {k}: " + ", ".join(f"{sp} {int(table[sp].sum())}" for sp in ("train", "val", "test")) +
              f"; classes with fewer than 2 test images: {int((table['test'] < 2).sum())}, without a training image: {int((table['train'] == 0).sum())}")
    print("mean RGB of the cache (sanity): ", np.load(cfg.HK_CACHE, mmap_mode='r')[::50].mean(axis=(0, 2, 3)).round(1))


if __name__ == "__main__":
    main()
