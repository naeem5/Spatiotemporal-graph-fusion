# look at the images once and write the study index (10 classes).
import pandas as pd

import config as cfg
import data_loading


def to_text(df):
    # DataFrame as text for the markdown report.
    try:
        return df.to_markdown()
    except Exception:
        return "```\n" + df.to_string() + "\n```"


def describe(index, title):
    # Markdown report of an index (printed and written to reports/).
    ok = index[index["ok"].astype(bool)]
    lines = [f"# {title}", "",
             f"Sequences in the table (after removing excluded classes): {len(index)}; readable with equal shape in all channels: {len(ok)}; "
             f"not usable (file missing, unreadable or channels differ): {len(index) - len(ok)}.", ""]
    for col in ("T", "Z", "H", "W", "dtype"):
        counts = ok[col].value_counts().sort_index() if col != "dtype" else ok[col].value_counts()
        lines.append(f"- {col}: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    if len(ok):
        lines += ["", f"Sequences with at least {cfg.MIN_T} frames (used by Stage 2, Stage 3 and the evaluation): "
                      f"{int((ok['T'] >= cfg.MIN_T).sum())} of {len(ok)} ({100 * (ok['T'] >= cfg.MIN_T).mean():.1f}%)."]
        if (ok["Z"] != cfg.Z_PLANES).any():
            lines.append(f"WARNING: not every stack has {cfg.Z_PLANES} z-planes; the others are resampled (Z_MODE = {cfg.Z_MODE}). Check that this is intended.")
        if ((ok["H"] != cfg.TARGET_HW) | (ok["W"] != cfg.TARGET_HW)).any():
            lines.append(f"Note: not every frame is {cfg.TARGET_HW} x {cfg.TARGET_HW} pixels; smaller frames are zero-padded, larger ones centre-cropped.")
        lines += ["", "## Sequences per class and split (all readable sequences)", "",
                  to_text(pd.crosstab(ok["label"], ok["split"], margins=True)), "",
                  f"## Sequences per class and split with at least {cfg.MIN_T} frames", "",
                  to_text(pd.crosstab(ok[ok["T"] >= cfg.MIN_T]["label"], ok[ok["T"] >= cfg.MIN_T]["split"], margins=True)), "",
                  "## Frames per sequence by class", "", to_text(pd.crosstab(ok["label"], ok["T"])), "",
                  "## Class names found in the table for each label (check the order against CLASS_NAMES)", "",
                  to_text(ok.groupby("label")["group"].agg(lambda s: s.value_counts().index[0]).to_frame())]
    bad = index[~index["ok"].astype(bool)]
    if len(bad):
        lines += ["", "## First unusable sequences", "", to_text(bad[["batch", "well", "field", "filename", "path_mito"]].head(15))]
    return "\n".join(lines)


def build(table, label_column, exclude, name, out_csv):
    if name == "ncm460" and cfg.SPLIT_MODE not in ("table", "field"):
        raise ValueError(f"SPLIT_MODE must be 'table' or 'field', not {cfg.SPLIT_MODE!r}")
    if name == "ncm460" and cfg.SPLIT_MODE == "field" and cfg.NCM_INDEX_TABLE.exists():
        # the shapes of the stacks are already known from the table-split index (reading 100,000 headers takes over an hour): only the split changes
        print(f"Re-using the image shapes of {cfg.NCM_INDEX_TABLE.name}; only the split is made again.")
        index = data_loading.resplit_by_field(pd.read_csv(cfg.NCM_INDEX_TABLE), cfg.SPLIT_FRACTIONS, cfg.SPLIT_SEED)
    else:
        index = data_loading.build_index(table, label_column, exclude, dataset_name=name)
        if name == "ncm460" and cfg.SPLIT_MODE == "field":
            index = data_loading.resplit_by_field(index, cfg.SPLIT_FRACTIONS, cfg.SPLIT_SEED)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out_csv, index=False)
    report = describe(index, f"{name}: study index")
    cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.REPORT_DIR / f"{name}_data_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten: {out_csv}")


def main():
    if not cfg.NCM_TABLE.exists():
        raise FileNotFoundError(f"Study table not found: {cfg.NCM_TABLE}. Copy NCM460_10class.csv into {cfg.NCM_TABLE.parent}.")
    build(cfg.NCM_TABLE, cfg.LABEL_COLUMN, cfg.EXCLUDE_LABELS, "ncm460", cfg.NCM_INDEX)
    if cfg.BUILD_A549_INDEX:
        if cfg.A549_TABLE.exists():
            build(cfg.A549_TABLE, "label", (), "a549", cfg.A549_INDEX)
        else:
            print(f"A549 table not found ({cfg.A549_TABLE}); the cross-dataset tests will not be possible.")


if __name__ == "__main__":
    main()

