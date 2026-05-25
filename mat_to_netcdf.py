"""
mat_to_netcdf.py
----------------
Converts all Strait of Georgia CODAR HF radar .mat files in a folder
into a single NetCDF file per year (e.g. sog_hfradar_2026.nc).

Reads all .mat files in DATA_DIR, merges them in chronological order,
and writes a clean CF-convention NetCDF to OUTPUT_DIR.

Usage:
    python mat_to_netcdf.py

Requirements:
    pip install scipy h5py numpy netCDF4
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\polinae\Downloads\Search Results (20260513T173040.877Z) - 36023943\search64576342"
OUTPUT_DIR = r"C:\Users\polinae\Documents\sog-hfradar\data\processed"

# Year to process — script will find all .mat files in DATA_DIR and filter by year
YEAR = 2026

# ── MATLAB datenum → Python datetime ─────────────────────────────────────────
def matlab_datenum_to_datetime(datenum):
    """
    Convert a MATLAB datenum (days since Jan 0, 0000) to a Python datetime (UTC).
    MATLAB epoch offset: 719529 days between MATLAB day 0 and Unix epoch (1970-01-01).
    """
    # Convert to Unix timestamp (seconds)
    unix_days = np.array(datenum, dtype=np.float64) - 719529.0
    unix_seconds = unix_days * 86400.0
    return unix_seconds  # Return as seconds since 1970-01-01 for NetCDF


def matlab_datenum_to_str(datenum):
    """Convert a single MATLAB datenum to a human-readable UTC string."""
    unix_seconds = (float(datenum) - 719529.0) * 86400.0
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Load a single .mat file ───────────────────────────────────────────────────
def load_mat(filepath):
    """Load .mat file, auto-detecting legacy vs HDF5 format."""
    with open(filepath, "rb") as f:
        header = f.read(128)

    hdf5_sig = b"\x89HDF\r\n\x1a\n"
    if hdf5_sig in header[:128]:
        raise NotImplementedError("HDF5 .mat files not yet supported in this script.")
    else:
        import scipy.io
        return scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)


# ── Extract data from one time step ──────────────────────────────────────────
def extract_timestep(ts_struct):
    """
    Pull U, V, QC flags, error estimates, and timestamp from one Totals struct entry.
    Returns a dict of 1-D arrays (length = n_grid_points).
    """
    return {
        "timestamp":   float(ts_struct.TimeStamp),
        "U":           np.array(ts_struct.U,    dtype=np.float32),
        "V":           np.array(ts_struct.V,    dtype=np.float32),
        "Uraw":        np.array(ts_struct.Uraw, dtype=np.float32),
        "Vraw":        np.array(ts_struct.Vraw, dtype=np.float32),
        "Uerr":        np.array(ts_struct.ErrorEstimates[0].Uerr,         dtype=np.float32),
        "Verr":        np.array(ts_struct.ErrorEstimates[0].Verr,         dtype=np.float32),
        "qc_overall":  np.array(ts_struct.QC.overallFlag,                 dtype=np.int8),
        "qc_speed":    np.array(ts_struct.QC.totalSpeedFlag,              dtype=np.int8),
        "qc_gdop":     np.array(ts_struct.QC.gdopFlag,                    dtype=np.int8),
        "n_rads":      np.array(ts_struct.OtherMatrixVars.makeTotals_TotalsNumRads, dtype=np.int8),
    }


# ── Load and merge all .mat files ────────────────────────────────────────────
def load_all_files(data_dir, year):
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    if not mat_files:
        print(f"ERROR: No .mat files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(mat_files)} .mat file(s):")
    for f in mat_files:
        print(f"  {os.path.basename(f)}")

    all_timesteps = []
    lonlat = None

    for filepath in mat_files:
        print(f"\nLoading: {os.path.basename(filepath)}")
        data = load_mat(filepath)
        totals = data["Totals"]  # shape (N,) object array

        print(f"  Time steps: {totals.shape[0]}")
        print(f"  First: {matlab_datenum_to_str(totals.flat[0].TimeStamp)}")
        print(f"  Last:  {matlab_datenum_to_str(totals.flat[-1].TimeStamp)}")

        # Grab grid from first file only (same for all)
        if lonlat is None:
            lonlat = np.array(totals.flat[0].LonLat, dtype=np.float64)  # (2532, 2)

        for ts in totals.flat:
            row = extract_timestep(ts)
            all_timesteps.append(row)

    print(f"\nTotal time steps across all files: {len(all_timesteps)}")

    # Sort by timestamp (in case files overlap or are out of order)
    all_timesteps.sort(key=lambda x: x["timestamp"])

    # Remove duplicates (same timestamp)
    seen = set()
    unique = []
    for ts in all_timesteps:
        t = ts["timestamp"]
        if t not in seen:
            seen.add(t)
            unique.append(ts)
    if len(unique) < len(all_timesteps):
        print(f"  Removed {len(all_timesteps) - len(unique)} duplicate time steps")

    print(f"Unique time steps: {len(unique)}")
    return unique, lonlat


# ── Write NetCDF ──────────────────────────────────────────────────────────────
def write_netcdf(timesteps, lonlat, output_path):
    import netCDF4 as nc

    n_time = len(timesteps)
    n_grid = lonlat.shape[0]

    # Stack arrays
    timestamps = np.array([ts["timestamp"] for ts in timesteps], dtype=np.float64)
    time_unix  = matlab_datenum_to_datetime(timestamps)  # seconds since 1970-01-01

    U         = np.stack([ts["U"]          for ts in timesteps])  # (n_time, n_grid)
    V         = np.stack([ts["V"]          for ts in timesteps])
    Uraw      = np.stack([ts["Uraw"]       for ts in timesteps])
    Vraw      = np.stack([ts["Vraw"]       for ts in timesteps])
    Uerr      = np.stack([ts["Uerr"]       for ts in timesteps])
    Verr      = np.stack([ts["Verr"]       for ts in timesteps])
    qc_overall= np.stack([ts["qc_overall"] for ts in timesteps])
    qc_speed  = np.stack([ts["qc_speed"]   for ts in timesteps])
    qc_gdop   = np.stack([ts["qc_gdop"]    for ts in timesteps])
    n_rads    = np.stack([ts["n_rads"]     for ts in timesteps])

    lons = lonlat[:, 0].astype(np.float32)
    lats = lonlat[:, 1].astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nWriting: {output_path}")
    with nc.Dataset(output_path, "w", format="NETCDF4") as ds:

        # ── Dimensions
        ds.createDimension("time",       n_time)
        ds.createDimension("grid_point", n_grid)

        # ── Global attributes (CF conventions)
        ds.title            = "Strait of Georgia CODAR HF Radar Surface Currents"
        ds.institution      = "Ocean Networks Canada"
        ds.source           = "CODAR SeaSonde HF Radar"
        ds.Conventions      = "CF-1.8"
        ds.history          = f"Created {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by mat_to_netcdf.py"
        ds.citation         = ("Ocean Networks Canada Society. 2016. Strait of Georgia "
                               "Oceanographic Radar System Deployed 2016-03-23. "
                               "https://doi.org/10.34943/10483644-44a9-486c-b9d8-a76fd466277e")
        ds.DOI              = "10.34943/10483644-44a9-486c-b9d8-a76fd466277e"
        ds.geospatial_lat_min = float(lats.min())
        ds.geospatial_lat_max = float(lats.max())
        ds.geospatial_lon_min = float(lons.min())
        ds.geospatial_lon_max = float(lons.max())
        ds.time_coverage_start = matlab_datenum_to_str(timestamps[0])
        ds.time_coverage_end   = matlab_datenum_to_str(timestamps[-1])
        ds.comment = ("QC flags: 1=good, 2=probably_good, 3=probably_bad, "
                      "4=bad, 9=missing. overallFlag is the primary QC variable.")

        # ── Coordinate variables
        t_var = ds.createVariable("time", "f8", ("time",), zlib=True)
        t_var.units         = "seconds since 1970-01-01 00:00:00 UTC"
        t_var.calendar      = "standard"
        t_var.standard_name = "time"
        t_var.long_name     = "Time (UTC)"
        t_var[:] = time_unix

        lon_var = ds.createVariable("lon", "f4", ("grid_point",), zlib=True)
        lon_var.units         = "degrees_east"
        lon_var.standard_name = "longitude"
        lon_var.long_name     = "Longitude of grid point"
        lon_var[:] = lons

        lat_var = ds.createVariable("lat", "f4", ("grid_point",), zlib=True)
        lat_var.units         = "degrees_north"
        lat_var.standard_name = "latitude"
        lat_var.long_name     = "Latitude of grid point"
        lat_var[:] = lats

        # ── Data variables
        fill = np.float32(np.nan)

        def make_var(name, long_name, units, data, dtype="f4"):
            v = ds.createVariable(name, dtype, ("time", "grid_point"),
                                  zlib=True, complevel=4,
                                  fill_value=np.float32(9.969209968386869e+36) if dtype == "f4" else -127)
            v.long_name = long_name
            v.units     = units
            v[:] = data
            return v

        u_var = make_var("U",    "Eastward surface current velocity (QC clean)", "cm s-1", U)
        u_var.standard_name = "eastward_sea_water_velocity"

        v_var = make_var("V",    "Northward surface current velocity (QC clean)", "cm s-1", V)
        v_var.standard_name = "northward_sea_water_velocity"

        make_var("U_raw",  "Eastward surface current velocity (raw)", "cm s-1", Uraw)
        make_var("V_raw",  "Northward surface current velocity (raw)", "cm s-1", Vraw)
        make_var("U_err",  "Eastward velocity error estimate (GDOP)", "cm s-1", Uerr)
        make_var("V_err",  "Northward velocity error estimate (GDOP)", "cm s-1", Verr)

        qc1 = ds.createVariable("qc_overall", "i1", ("time", "grid_point"),
                                 zlib=True, complevel=4, fill_value=np.int8(-127))
        qc1.long_name    = "Overall QC flag"
        qc1.flag_values  = "1 2 3 4 9"
        qc1.flag_meanings = "good probably_good probably_bad bad missing"
        qc1[:] = qc_overall

        qc2 = ds.createVariable("qc_speed", "i1", ("time", "grid_point"),
                                  zlib=True, complevel=4, fill_value=np.int8(-127))
        qc2.long_name = "Speed QC flag"
        qc2[:] = qc_speed

        qc3 = ds.createVariable("qc_gdop", "i1", ("time", "grid_point"),
                                  zlib=True, complevel=4, fill_value=np.int8(-127))
        qc3.long_name = "GDOP QC flag"
        qc3[:] = qc_gdop

        nr = ds.createVariable("n_radials", "i1", ("time", "grid_point"),
                                zlib=True, complevel=4, fill_value=np.int8(-127))
        nr.long_name = "Number of radials used in total vector"
        nr[:] = n_rads

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
    print(f"\nSummary:")
    print(f"  Time steps : {n_time}")
    print(f"  Grid points: {n_grid}")
    print(f"  Start      : {matlab_datenum_to_str(timestamps[0])}")
    print(f"  End        : {matlab_datenum_to_str(timestamps[-1])}")
    print(f"  Lon range  : {lons.min():.4f} → {lons.max():.4f}")
    print(f"  Lat range  : {lats.min():.4f} → {lats.max():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_path = os.path.join(OUTPUT_DIR, f"sog_hfradar_{YEAR}.nc")

    timesteps, lonlat = load_all_files(DATA_DIR, YEAR)
    write_netcdf(timesteps, lonlat, output_path)

    print(f"\nNetCDF written to:\n  {output_path}")
    print("\nTo open in Python later:")
    print("  import xarray as xr")
    print(f"  ds = xr.open_dataset(r'{output_path}')")
    print("  ds_good = ds.where(ds.qc_overall == 1)  # apply QC mask")
