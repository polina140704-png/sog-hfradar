# sog-hfradar
Analysis of surface ocean currents in the Strait of Georgia using CODAR HF radar data from Ocean Networks Canada (ONC).

## Data
- Source: ONC Oceans 3.0 API (SOGCS, CODARQCSC)
- Coverage: 2016–2026 (in progress)
- Resolution: hourly, ~1km grid

## Scripts
- `download_onc_data.py` - downloads NetCDF data from ONC API by chunk
- `explore_hfradar.py` - explores raw .mat file structure
- `mat_to_netcdf.py` - converts .mat files to NetCDF format

## Notebooks
- `01_explore_2026.ipynb` - initial data exploration and availability plots

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements
See `requirements.txt`
