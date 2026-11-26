# tile sequences and study index of the Live-cell Phase Contrast data set (Forsgren et al., DOI 10.17044/scilifelab.29315642).
import io
import json
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

import config as cfg

_SRC = None


class Source:
    # Read access to the files of one cell line, from <root>/<cell>.zip or from the unzipped folder <root>/<cell>/.

    def __init__(self, root, cell):
        root = Path(root)
        self.cell = cell
        if (root / f"{cell}.zip").exists():
            self.zip = zipfile.ZipFile(root / f"{cell}.zip")
            self.names = set(self.zip.namelist())
        elif (root / cell).is_dir():
            self.zip = None
            self.base = root
            self.names = {str(Path(dp, f).relative_to(root)).replace("\\", "/") for dp, _, fs in os.walk(root / cell) for f in fs}
        else:
            raise FileNotFoundError(f"neither {root / (cell + '.zip')} nor the folder {root / cell} exists (cfg.LCTP_ROOT)")

    def read_tif(self, name):
        data = self.zip.read(name) if self.zip is not None else (self.base / name).read_bytes()
        return tifffile.imread(io.BytesIO(data))


def _init_worker(root, cell):
    global _SRC
    _SRC = Source(root, cell)


def scan_table(src):
    # {vessel id: {datetime: folder prefix of that scan}} from the names of the phase-contrast images.
    rx = re.compile(rf"^{re.escape(src.cell)}/(?:ScanData/)?(\d{{4}})/(\d{{2}})/(\d{{4}})/(\d+)/[A-P]\d+-1-Ph\.tif$")
    scans = {}
    for name in src.names:
        m = rx.match(name)
        if m:
            yymm, dd, hhmm, vessel = m.groups()
            dt = datetime(2000 + int(yymm[:2]), int(yymm[2:]), int(dd), int(hhmm[:2]), int(hhmm[2:]))
            scans.setdefault(int(vessel), {}).setdefault(dt, name.rsplit("/", 1)[0])       # duplicates of a scan: the first copy is used
    return scans


def mask_name(src, job, dt, well):
    stamp = f"{dt:%y}y{dt:%m}m{dt:%d}d{dt:%H}h{dt:%M}"
    for pre in (f"{src.cell}/Jobs/{job}", f"{src.cell}/{job}"):
        name = f"{pre}/{stamp}/{well}-1-AichPhLabels.tif"
        if name in src.names:
            return name
    return None


def pick_frames(times):
    # Scans nearest to cfg.WINDOW_HOURS (within MAX_SCAN_DEVIATION_H) -> [(wanted hour, datetime)], and the position of the anchor frame.
    t0 = min(times)
    out = []
    for h in cfg.WINDOW_HOURS:
        best = min(times, key=lambda t: abs((t - t0).total_seconds() / 3600 - h))
        if abs((best - t0).total_seconds() / 3600 - h) <= cfg.MAX_SCAN_DEVIATION_H:
            out.append((h, best))
    anchor = [i for i, (h, _) in enumerate(out) if h == cfg.ANCHOR_HOUR]
    return out, (anchor[0] if anchor else None)


def process_well(job):
    # Cut the tiles of one well: choose tiles at the anchor frame, read the window frames once, write the tile arrays. Returns the index rows.
    src = _SRC
    frames, anchor = job["frames"], job["anchor"]
    rng = np.random.default_rng([cfg.TILE_SEED, job["seed"]])
    T = cfg.TILE
    anchor_img = src.read_tif(job["frame_names"][anchor])
    rows, cols = anchor_img.shape[0] // T, anchor_img.shape[1] // T
    counts = None
    if job["mask"]:
        mask = src.read_tif(job["mask"])
        counts = {(r, c): len(np.unique(mask[r * T:(r + 1) * T, c * T:(c + 1) * T])) - (1 if (mask[r * T:(r + 1) * T, c * T:(c + 1) * T] == 0).any() else 0)
                  for r in range(rows) for c in range(cols)}
    cells = [(r, c) for r in range(rows) for c in range(cols) if counts is None or counts[(r, c)] >= cfg.MIN_CELLS_PER_TILE]
    k = min(job["n_tiles"], len(cells))
    chosen = [cells[i] for i in rng.choice(len(cells), size=k, replace=False)] if k else []
    stack = np.zeros((len(frames), len(chosen), T, T), dtype=np.uint8)
    for i, name in enumerate(job["frame_names"]):
        img = anchor_img if i == anchor else src.read_tif(name)
        for j, (r, c) in enumerate(chosen):
            stack[i, j] = img[r * T:(r + 1) * T, c * T:(c + 1) * T]
    out = []
    folder = Path(job["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    for j, (r, c) in enumerate(chosen):
        path = folder / f"{job['well']}_r{r}c{c}.npy"
        np.save(path, stack[:, j])
        out.append(dict(path=str(path), row=r, col=c, T=len(frames), anchor=anchor, n_cells_anchor=(counts[(r, c)] if counts else -1)))
    return job["key"], out


def make_splits(wells, only_wells):
    # Add the columns split_plate_f<k> and split_compound_f<k> to the well table (one row per well).
    wells = wells.copy()
    plates = list(dict.fromkeys(wells.sort_values("plate_start")["plate"]))
    cmpds = {m: sorted(g["compound"].unique()) for m, g in wells[wells["moa"] != "DMSO"].groupby("moa")}
    if only_wells:                                              # code test: wells are dealt out cyclically, nothing else is meaningful
        for f in range(max(len(plates), 1)):
            wells[f"split_plate_f{f}"] = [("train", "val", "test")[i % 3] for i in range(len(wells))]
        for f in range(cfg.COMPOUND_FOLDS):
            wells[f"split_compound_f{f}"] = [("train", "val", "test")[(i + f) % 3] for i in range(len(wells))]
        return wells
    rng = np.random.default_rng(cfg.TILE_SEED)
    order = {m: [c for c in np.array(cs)[rng.permutation(len(cs))]] for m, cs in cmpds.items()}      # one fixed shuffle per class, shared by all folds
    dmso_wells = {p: list(np.array(sorted(g["well"]))[rng.permutation(len(g))]) for p, g in wells[wells["moa"] == "DMSO"].groupby("plate")}
    for f, test_plate in enumerate(plates):                     # ---- plate protocol ----
        col = pd.Series("train", index=wells.index)
        col[wells["plate"] == test_plate] = "test"
        for m, cs in order.items():
            val = cs[(f * cfg.VAL_COMPOUNDS_PER_CLASS):(f * cfg.VAL_COMPOUNDS_PER_CLASS) + cfg.VAL_COMPOUNDS_PER_CLASS]
            col[(wells["moa"] == m) & wells["compound"].isin(val) & (wells["plate"] != test_plate)] = "val"
        for p, ws in dmso_wells.items():
            if p != test_plate:
                col[(wells["plate"] == p) & wells["well"].isin(ws[:cfg.VAL_DMSO_WELLS_PER_PLATE])] = "val"
        wells[f"split_plate_f{f}"] = col
    n_test = len(next(iter(order.values()))) // cfg.COMPOUND_FOLDS
    for f in range(cfg.COMPOUND_FOLDS):                         # ---- compound protocol ----
        col = pd.Series("train", index=wells.index)
        for m, cs in order.items():
            test = cs[f * n_test:(f + 1) * n_test]
            rest = [c for c in cs if c not in test]
            val = rest[(f * cfg.VAL_COMPOUNDS_PER_CLASS) % len(rest):][:cfg.VAL_COMPOUNDS_PER_CLASS]
            col[(wells["moa"] == m) & wells["compound"].isin(test)] = "test"
            col[(wells["moa"] == m) & wells["compound"].isin(val)] = "val"
        for p, ws in dmso_wells.items():
            chunks = np.array_split(np.arange(len(ws)), cfg.COMPOUND_FOLDS)
            test_w = [ws[i] for i in chunks[f]]
            val_w = [ws[i] for i in chunks[(f + 1) % cfg.COMPOUND_FOLDS]][:cfg.VAL_DMSO_WELLS_PER_PLATE]
            col[(wells["plate"] == p) & wells["well"].isin(test_w)] = "test"
            col[(wells["plate"] == p) & wells["well"].isin(val_w)] = "val"
        wells[f"split_compound_f{f}"] = col
    return wells


def main():
    cell = cfg.CELL_LINE
    if cell is None:
        raise SystemExit("set PUBLIC_DATASET to lctp_a549 or lctp_u2os")
    pm = pd.read_csv(cfg.PLATE_MAP)
    pm = pm[(pm["Metadata_Cells"] == cell) & pm["Metadata_cmpd_moa_group"].notna()].copy()
    pm["moa"] = pm["Metadata_cmpd_moa_group"].str.upper().replace({"DMSO": "DMSO"})
    only = [w.strip() for w in os.environ.get("PUBLIC_LCTP_WELLS", "").split(",") if w.strip()]
    if only:
        pm = pm[pm["Metadata_Well"].isin(only)]
    classes = cfg.LCTP_CLASS_ORDER
    assert set(pm["moa"]) <= set(classes), f"unexpected groups: {set(pm['moa']) - set(classes)}"

    src = Source(cfg.LCTP_ROOT, cell)
    scans = scan_table(src)
    print(f"{cell}: scans per vessel " + ", ".join(f"{v}: {len(t)}" for v, t in sorted(scans.items())))
    jobs, plate_start, notes = [], {}, []
    for (plate, vessel, job_id), g in pm.groupby(["Metadata_Plate", "Metadata_VesselID", "Metadata_AIch_job"]):
        times = sorted(scans.get(int(vessel), {}))
        if not times:
            notes.append(f"plate {plate} (vessel {vessel}) has no scans in the data: skipped")
            continue
        frames, anchor = pick_frames(times)
        if anchor is None or len(frames) < cfg.MIN_T:
            notes.append(f"plate {plate}: no frame at {cfg.ANCHOR_HOUR} h or fewer than {cfg.MIN_T} frames: skipped")
            continue
        plate_start[plate] = times[0]
        notes.append(f"plate {plate} (vessel {vessel}, job {job_id}): {len(times)} scans, {len(frames)} window frames, anchor frame {anchor} "
                     f"({frames[anchor][1]:%Y-%m-%d %H:%M})")
        for _, w in g.iterrows():
            well = w["Metadata_Well"]
            names = [f"{scans[int(vessel)][dt]}/{well}-1-Ph.tif" for _, dt in frames]
            if not all(n in src.names for n in names):
                notes.append(f"  well {well}: images missing, skipped")
                continue
            n_tiles = cfg.TILES_PER_DMSO_WELL if w["moa"] == "DMSO" else cfg.TILES_PER_WELL
            jobs.append(dict(key=(plate, well), well=well, frames=frames, frame_names=names, anchor=anchor, n_tiles=n_tiles,
                             mask=mask_name(src, int(job_id), frames[anchor][1], well),
                             seed=int.from_bytes(f"{plate}|{well}".encode(), "little") % (2 ** 31),          # stable between runs (str hash() is not)
                             out_dir=str(cfg.TILE_CACHE / plate)))
    print("\n".join(notes))
    print(f"{len(jobs)} wells; cutting tiles with {cfg.BUILD_WORKERS} worker(s)")

    results = {}
    if cfg.BUILD_WORKERS <= 1:
        _init_worker(cfg.LCTP_ROOT, cell)
        for job in jobs:
            k, rows = process_well(job)
            results[k] = rows
    else:
        with ProcessPoolExecutor(cfg.BUILD_WORKERS, initializer=_init_worker, initargs=(cfg.LCTP_ROOT, cell)) as ex:
            for n, (k, rows) in enumerate(ex.map(process_well, jobs, chunksize=2), 1):
                results[k] = rows
                if n % 50 == 0:
                    print(f"  {n}/{len(jobs)} wells")

    wells = pm.rename(columns={"Metadata_Plate": "plate", "Metadata_Well": "well", "Metadata_cmpd_cmpdname": "compound"})[["plate", "well", "compound", "moa"]].copy()
    wells["compound"] = wells["compound"].astype(str).str.replace("\xa0", " ").str.strip()
    wells = wells[[(p, w) in results for p, w in zip(wells["plate"], wells["well"])]].copy()
    wells["plate_start"] = wells["plate"].map(plate_start)
    wells = make_splits(wells, bool(only))
    wells["label"] = wells["moa"].map({m: i for i, m in enumerate(classes)})

    rows = []
    for _, w in wells.iterrows():
        for t in results[(w["plate"], w["well"])]:
            path = Path(t["path"])
            rel = path.relative_to(cfg.PUB_DIR) if path.is_relative_to(cfg.PUB_DIR) else path
            rows.append({**t, "path": str(rel).replace("\\", "/"), "dataset": cfg.DATASET, "cell_line": cell, "plate": w["plate"], "well": f"{w['plate']}|{w['well']}",
                         "compound": w["compound"], "moa": w["moa"], "label": int(w["label"]),
                         **{c: w[c] for c in wells.columns if c.startswith("split_")}})
    df = pd.DataFrame(rows)
    cfg.CSV_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.INDEX, index=False)
    counts = df.groupby("label").size().reindex(range(len(classes)), fill_value=0)
    cfg.CLASSES_JSON.write_text(json.dumps(dict(names=classes, control_label=0, counts=counts.tolist()), indent=1), encoding="utf-8")

    lines = [f"# {cfg.DATASET} data report", "", *notes, "", f"Tiles: {len(df)}; wells: {df['well'].nunique()}; compounds: {df['compound'].nunique()}", "", "Tiles per class:", "```",
             pd.DataFrame(dict(name=classes, tiles=counts.values)).to_string(), "```", ""]
    for col in [c for c in df.columns if c.startswith("split_")]:
        lines += [f"## {col}", "", "```", df.groupby(col).agg(tiles=("path", "size"), wells=("well", "nunique"), compounds=("compound", "nunique")).to_string(), "```", ""]
    cfg.REPORT_BASE.mkdir(parents=True, exist_ok=True)
    (cfg.REPORT_BASE / f"{cfg.DATASET}_data_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"\nwritten: {cfg.INDEX}, {cfg.CLASSES_JSON}")


if __name__ == "__main__":
    main()
