"""
explore_hfradar.py
------------------
Initial exploration of Strait of Georgia CODAR HF radar .mat files.
Handles both legacy .mat (scipy) and v7.3 HDF5 .mat (h5py) formats.

Usage:
    python explore_hfradar.py

Requirements:
    pip install scipy h5py numpy
"""

import os
import sys
import numpy as np

# ── File paths ────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\polinae\Downloads\Search Results (20260513T173040.877Z) - 36023943\search64576342"

FILE_1 = os.path.join(
    DATA_DIR,
    "StraitofGeorgia_StraitofGeorgiaCODARSystem_OceanographicRadarSystem"
    "_20260101T000000.000Z_20260301T090000.000Z-Totals_Clean.mat",
)
FILE_2 = os.path.join(
    DATA_DIR,
    "StraitofGeorgia_StraitofGeorgiaCODARSystem_OceanographicRadarSystem"
    "_20260301T100000.000Z_20260429T180000.000Z-Totals_Clean.mat",
)

TARGET_FILE = FILE_1  # change to FILE_2 to explore the second file


# ── Loader: auto-detects format ───────────────────────────────────────────────
def load_mat(filepath):
    """
    Load a .mat file, auto-detecting v5/v6 (scipy) vs v7.3 HDF5 (h5py).
    Returns (data, format_str).
    """
    import struct

    with open(filepath, "rb") as f:
        header = f.read(128)

    # v7.3 files start with the HDF5 signature at byte 1 (after a description)
    hdf5_sig = b"\x89HDF\r\n\x1a\n"
    if hdf5_sig in header[:128]:
        print("  → Detected v7.3 HDF5 format, using h5py")
        import h5py
        data = h5py.File(filepath, "r")
        return data, "hdf5"
    else:
        print("  → Detected legacy format, using scipy.io")
        import scipy.io
        data = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return data, "legacy"


# ── Recursive structure printer ───────────────────────────────────────────────
def print_hdf5_structure(obj, indent=0, max_depth=4):
    """Recursively print HDF5 group/dataset structure."""
    import h5py
    prefix = "  " * indent
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        for key in obj.keys():
            item = obj[key]
            if isinstance(item, h5py.Dataset):
                print(f"{prefix}📊 {key}  shape={item.shape}  dtype={item.dtype}")
                # Print a small sample
                if item.size > 0 and indent < max_depth:
                    try:
                        sample = item[()] if item.size <= 5 else item.flat[:5]
                        print(f"{prefix}   sample: {np.array(sample).ravel()[:5]}")
                    except Exception:
                        pass
            elif isinstance(item, h5py.Group):
                print(f"{prefix}📁 {key}/")
                if indent < max_depth:
                    print_hdf5_structure(item, indent + 1, max_depth)
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}📊 (dataset) shape={obj.shape}  dtype={obj.dtype}")


def print_legacy_structure(data, indent=0, max_depth=3):
    """Recursively print scipy mat_struct / ndarray structure."""
    import scipy.io
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, val in data.items():
            if key.startswith("__"):
                continue
            _describe(key, val, prefix, indent, max_depth)
    elif isinstance(data, scipy.io.matlab.mio5_params.MatlabObject):
        for key in data._fieldnames:
            val = getattr(data, key)
            _describe(key, val, prefix, indent, max_depth)


def _describe(key, val, prefix, indent, max_depth):
    import scipy.io
    if isinstance(val, np.ndarray):
        print(f"{prefix}📊 {key}  shape={val.shape}  dtype={val.dtype}")
        if val.size > 0 and val.dtype.kind in "fiu":
            flat = val.ravel()
            sample = flat[:5]
            print(f"{prefix}   sample: {sample}  |  min={flat.min():.4g}  max={flat.max():.4g}")
    elif isinstance(val, scipy.io.matlab.mio5_params.MatlabObject):
        print(f"{prefix}📁 {key}/ (struct)")
        if indent < max_depth:
            print_legacy_structure(val, indent + 1, max_depth)
    else:
        print(f"{prefix}  {key}: {type(val).__name__} = {val}")


# ── Main exploration ──────────────────────────────────────────────────────────
def explore(filepath):
    print("=" * 70)
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"SIZE: {os.path.getsize(filepath) / 1e6:.2f} MB")
    print("=" * 70)

    data, fmt = load_mat(filepath)

    print("\n── TOP-LEVEL VARIABLES ─────────────────────────────────────────────\n")
    if fmt == "hdf5":
        print_hdf5_structure(data)

        # Try to print time coverage from common CODAR variable names
        print("\n── TIME INFO (if present) ──────────────────────────────────────────")
        for tkey in ["TimeStamp", "time", "Time", "t"]:
            if tkey in data:
                ts = data[tkey][()]
                print(f"  {tkey}: shape={ts.shape}, first={ts.ravel()[0]}, last={ts.ravel()[-1]}")

        # Print lat/lon extents
        print("\n── SPATIAL EXTENT (if present) ─────────────────────────────────────")
        for pair in [("Lon", "Lat"), ("lon", "lat"), ("longitude", "latitude"), ("LON", "LAT")]:
            lk, latk = pair
            if lk in data and latk in data:
                lo = data[lk][()]
                la = data[latk][()]
                print(f"  Lon: {lo.min():.4f} → {lo.max():.4f}")
                print(f"  Lat: {la.min():.4f} → {la.max():.4f}")
                break

    else:  # legacy
        print_legacy_structure(data)

        print("\n── TIME INFO (if present) ──────────────────────────────────────────")
        for tkey in ["TimeStamp", "time", "Time", "t"]:
            if tkey in data:
                ts = np.atleast_1d(data[tkey])
                print(f"  {tkey}: shape={ts.shape}, first={ts.ravel()[0]}, last={ts.ravel()[-1]}")

        print("\n── SPATIAL EXTENT (if present) ─────────────────────────────────────")
        for lk, latk in [("Lon", "Lat"), ("lon", "lat"), ("longitude", "latitude")]:
            if lk in data and latk in data:
                lo = np.atleast_1d(data[lk]).ravel()
                la = np.atleast_1d(data[latk]).ravel()
                print(f"  Lon: {lo.min():.4f} → {lo.max():.4f}")
                print(f"  Lat: {la.min():.4f} → {la.max():.4f}")
                break

    print("\n── DONE ─────────────────────────────────────────────────────────────\n")
    return data, fmt


def explore_struct(s, name="", indent=0):
    """Recursively unpack a scipy mat_struct and print its fields."""
    import scipy.io
    prefix = "  " * indent
    if hasattr(s, '_fieldnames'):
        print(f"{prefix}📁 {name}/")
        for field in s._fieldnames:
            val = getattr(s, field)
            explore_struct(val, field, indent + 1)
    elif isinstance(s, np.ndarray):
        if s.dtype == object and s.size > 0:
            # Array of structs — inspect first element
            print(f"{prefix}📦 {name}  [{s.shape} object array] — inspecting [0]:")
            explore_struct(s.flat[0], f"{name}[0]", indent + 1)
        else:
            flat = s.ravel()
            if s.dtype.kind in "fiu" and s.size > 0:
                print(f"{prefix}📊 {name}  shape={s.shape}  dtype={s.dtype}  "
                      f"min={flat.min():.4g}  max={flat.max():.4g}  sample={flat[:3]}")
            elif s.dtype.kind in "USO" and s.size > 0:
                print(f"{prefix}📝 {name}  shape={s.shape}  sample={flat[:3]}")
            else:
                print(f"{prefix}  {name}  shape={s.shape}  dtype={s.dtype}")
    else:
        print(f"{prefix}  {name}: {type(s).__name__} = {s}")


if __name__ == "__main__":
    if not os.path.exists(TARGET_FILE):
        print(f"ERROR: File not found:\n  {TARGET_FILE}")
        print("Check DATA_DIR path at the top of this script.")
        sys.exit(1)

    data, fmt = explore(TARGET_FILE)

    if fmt == "legacy":
        print("\n── DEEP DIVE: Totals struct ────────────────────────────────────────\n")
        explore_struct(data["Totals"], "Totals")

        print("\n── DEEP DIVE: Grid struct ──────────────────────────────────────────\n")
        explore_struct(data["Grid"], "Grid")

        print("\n── DEEP DIVE: Meta struct ──────────────────────────────────────────\n")
        explore_struct(data["Meta"], "Meta")
