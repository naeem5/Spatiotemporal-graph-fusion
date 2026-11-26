# reading image sequences (TIFF for NCM460, NPZ for A549) and putting them into the shape the models expect.
import re
import zipfile
from pathlib import Path

import numpy as np
import tifffile

import config as cfg
from pdgraph.data import fit_hw as fit_hw_to_size
from pdgraph.data import robust_norm  # noqa: F401  (re-exported for the scripts)


# ------------------------------------------------------------------------------------------------------
# paths
# ------------------------------------------------------------------------------------------------------
def resolve_path(table_path):
    # Return the file that belongs to a path written in a study table.
    p = str(table_path).replace("\\", "/")
    candidates = [Path(p)]
    for old, new in cfg.PATH_PREFIX_MAP:
        if p.startswith(old):
            candidates.append(Path(str(new)) / p[len(old):].lstrip("/"))
    tail = re.search(r"(ncm460-p\d+/.*)$", p)
    if tail:
        for sub in ("",):
            candidates.append(Path(cfg.DATA_ROOT) / sub / tail.group(1))
    for c in candidates:
        if c.exists():
            return c
    return candidates[1] if len(candidates) > 1 else candidates[0]


def channel_path(mito_path, channel):
    # Path of the file of another channel of the same sequence (the channel name is a folder: .../mito/Well.../n.tif).
    return str(mito_path).replace("/mito/", f"/{channel}/").replace("\\mito\\", f"\\{channel}\\")


# ------------------------------------------------------------------------------------------------------
# shapes without reading the pixels
# ------------------------------------------------------------------------------------------------------
def _npz_shape(path):
    # Shape and dtype of the first array of an .npz read from its .npy header (the pixels are not loaded).
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        name = "img.npy" if "img.npy" in names else names[0]
        with zf.open(name) as f:
            version = np.lib.format.read_magic(f)
            shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f))
    return tuple(shape), str(dtype)


def probe(path):
    # Return (T, Z, H, W, dtype) of a sequence file, or None when the file is missing, unreadable or has an unexpected layout.
    path = Path(path)
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            with tifffile.TiffFile(str(path)) as tf:
                series = tf.series[0]
                shape, dtype = tuple(series.shape), str(series.dtype)
        elif path.suffix.lower() == ".npz":
            shape, dtype = _npz_shape(path)
        else:
            return None
    except Exception:
        return None
    if len(shape) == 3:                       # (Z, H, W): one frame
        shape = (1,) + shape
    if len(shape) != 4:
        return None
    return (*shape, dtype)


# ------------------------------------------------------------------------------------------------------
# pixels
# ------------------------------------------------------------------------------------------------------
def read_sequence(path):
    # Whole sequence as an array (T, Z, H, W).
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(str(path))
    else:
        d = np.load(path)
        arr = d["img"] if "img" in d else d[d.files[0]]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise ValueError(f"unexpected array shape {arr.shape} in {path}")
    return arr


def read_frame(path, t, z_actual):
    # One frame (Z, H, W) of a sequence. TIFF pages are stored frame after frame (page = t * Z + z), so only the pages of the
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(str(path)) as tf:
                series = tf.series[0]
                if len(series.shape) == 4 and series.shape[1] == z_actual:
                    return np.asarray(tf.asarray(key=list(range(t * z_actual, (t + 1) * z_actual))))
                if len(series.shape) == 3 and t == 0:
                    return np.asarray(series.asarray())
        except Exception:
            pass
    return read_sequence(path)[t]


# ------------------------------------------------------------------------------------------------------
# shape handling
# ------------------------------------------------------------------------------------------------------
def match_z(stack, z_planes=None):
    # (Z, H, W) or (T, Z, H, W) -> the same with exactly z_planes planes (uniformly chosen planes; unchanged when Z already fits).
    z_planes = z_planes or cfg.Z_PLANES
    axis = stack.ndim - 3
    z = stack.shape[axis]
    if z == z_planes:
        return stack
    idx = np.linspace(0, z - 1, z_planes).round().astype(int)       # fewer planes: nearest planes are repeated
    return np.take(stack, idx, axis=axis)


def fit_hw(arr, target=None):
    return fit_hw_to_size(arr, target or cfg.TARGET_HW)


