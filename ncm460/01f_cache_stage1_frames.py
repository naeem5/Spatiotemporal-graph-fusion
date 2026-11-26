# store the middle frame of every training and validation sequence on the local disk (for Stage 1).
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import config as cfg
import data_loading
import image_io

FRAME_SHAPE = (len(cfg.CHANNELS), cfg.Z_PLANES, cfg.TARGET_HW, cfg.TARGET_HW)
SAVE_EVERY = 512                                           # rows between saves of the progress mask


def read_middle_frame(rec):
    # Middle frame of a sequence as uint16 (3, 27, 144, 144).
    t = int(rec["T"]) // 2
    planes = [image_io.match_z(image_io.read_frame(rec[f"path_{c}"], t, int(rec["Z"]))) for c in cfg.CHANNELS]
    return image_io.fit_hw(np.stack(planes)).astype(np.uint16)


def cache_split(split):
    records = data_loading.load_index(cfg.NCM_INDEX, split, min_T=1)
    cfg.STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames_path, done_path = cfg.STAGE1_CACHE_DIR / f"{split}_frames.npy", cfg.STAGE1_CACHE_DIR / f"{split}_done.npy"
    np.save(cfg.STAGE1_CACHE_DIR / f"{split}_labels.npy", np.array([int(r["label"]) for r in records]))
    if frames_path.exists() and done_path.exists():
        frames, done = np.load(frames_path, mmap_mode="r+"), np.load(done_path)
        if frames.shape != (len(records),) + FRAME_SHAPE or len(done) != len(records):
            raise ValueError(f"{frames_path} does not fit the current index; delete the cache folder and start again")
    else:
        frames = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint16, shape=(len(records),) + FRAME_SHAPE)
        done = np.zeros(len(records), dtype=bool)
    todo = [i for i in np.argsort([r["path_mito"] for r in records]) if not done[i]]
    print(f"[{split}] {len(records)} sequences, {len(todo)} still to read")

    def work(i):
        frames[i] = read_middle_frame(records[i])
        return i

    with ThreadPoolExecutor(cfg.CACHE_WORKERS) as pool:
        for n, i in enumerate(tqdm(pool.map(work, todo), total=len(todo), desc=f"caching {split} frames"), start=1):
            done[i] = True
            if n % SAVE_EVERY == 0:
                frames.flush()
                np.save(done_path, done)
    frames.flush()
    np.save(done_path, done)
    return records, frames, done


def main():
    if cfg.SPLIT_MODE != "table":
        raise ValueError("the Stage 1 cache is made for the table split only (SPLIT_MODE = 'table'): there is no room for a second copy")
    n_train = len(data_loading.load_index(cfg.NCM_INDEX, "train", min_T=1))
    n_val = len(data_loading.load_index(cfg.NCM_INDEX, "val", min_T=1))
    need = (n_train + n_val) * int(np.prod(FRAME_SHAPE)) * 2
    cfg.STAGE1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cfg.STAGE1_CACHE_DIR).free
    print(f"about {need / 1e9:.0f} GB for {n_train} train and {n_val} validation frames in {cfg.STAGE1_CACHE_DIR}; free space: {free / 1e9:.0f} GB")
    already = sum(getattr(p.stat(), "st_blocks", 0) * 512 for p in cfg.STAGE1_CACHE_DIR.glob("*_frames.npy"))      # space really used (the arrays are sparse files)
    if free + already < need * 1.03:
        raise OSError("not enough free space for the Stage 1 cache; free some space or choose STAGE1_CACHE_DIR on a larger disk")

    start = time.time()
    results = {split: cache_split(split) for split in ("train", "val")}
    print(f"caching took {(time.time() - start) / 60:.1f} min")

    # ---- checks: all rows done, 5 random rows equal the images, random-read speed ----
    rng = np.random.default_rng(0)
    for split, (records, frames, done) in results.items():
        if not done.all():
            raise RuntimeError(f"{split}: {int((~done).sum())} rows are missing; run the script again")
        for i in rng.choice(len(records), size=5, replace=False):
            if not np.array_equal(frames[i], read_middle_frame(records[i])):
                raise RuntimeError(f"{split}: cached row {i} differs from the images")
    frames = results["train"][1]
    rows = rng.choice(len(frames), size=min(200, len(frames)), replace=False)
    t0 = time.time()
    for i in rows:
        _ = np.array(frames[i])
    speed = len(rows) / (time.time() - t0)
    print(f"cache complete and verified. Random read speed of the cache: {speed:.0f} frames/s (about 190 are needed; the operating-system cache may flatter this number).")


if __name__ == "__main__":
    main()
