# read every NCM460 sequence once and store its z-projection (the only thing Stage 2 and Stage 3 use of the graph input).
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import config as cfg
import data_loading
import image_io


def cache_one(rec):
    # Project one sequence and save it (through a temporary name, so a stopped run never leaves a half-written file).
    out = data_loading.graph_cache_path(rec)
    if out.exists():
        return 0
    proj = data_loading.project_sequences([image_io.read_sequence(rec[f"path_{c}"]) for c in cfg.CHANNELS])
    tmp = out.with_name(out.stem + ".tmp.npy")
    np.save(tmp, proj)
    os.replace(tmp, out)
    return 1


def main():
    records = data_loading.load_index(cfg.NCM_INDEX_TABLE, None, min_T=cfg.MIN_T)
    records.sort(key=lambda r: r["path_mito"])                 # path order = roughly disk order
    cfg.GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    need = sum(int(r["T"]) * len(cfg.CHANNELS) * cfg.TARGET_HW ** 2 * 2 for r in records)
    free = shutil.disk_usage(cfg.GRAPH_CACHE_DIR).free
    print(f"{len(records)} sequences with at least {cfg.MIN_T} frames; about {need / 1e9:.1f} GB in {cfg.GRAPH_CACHE_DIR}; free space: {free / 1e9:.0f} GB")
    if free < need * 1.1:
        raise OSError("not enough free space for the cache; set GRAPH_CACHE_DIR in config.py to a folder on a larger disk")

    start, written = time.time(), 0
    with ThreadPoolExecutor(cfg.CACHE_WORKERS) as pool:
        for n in tqdm(pool.map(cache_one, records), total=len(records), desc="caching projections"):
            written += n
    print(f"{written} sequences written, {len(records) - written} were already cached, {(time.time() - start) / 60:.1f} min")

    # ---- checks: everything present, and 5 random sequences equal a fresh computation ----
    missing = [r for r in records if not data_loading.graph_cache_path(r).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} sequences are not cached; run the script again")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(records), size=min(5, len(records)), replace=False):
        rec = records[i]
        fresh = data_loading.project_sequences([image_io.read_sequence(rec[f"path_{c}"]) for c in cfg.CHANNELS])
        if not np.array_equal(fresh, np.load(data_loading.graph_cache_path(rec))):
            raise RuntimeError(f"cached projection differs from the images for {rec['path_mito']}")
    print("Cache complete and verified. Stage 2 (03_train_stage2_graph.py) can start.")


if __name__ == "__main__":
    main()
