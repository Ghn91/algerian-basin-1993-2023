# Data Access Instructions

Raw NetCDF files are NOT included in this repository due to size (~30 GB total).
All data are freely available from the Copernicus Marine Service.

## Download Instructions

### 1. Create a CMEMS account
Register at: https://marine.copernicus.eu (free)

### 2. Install the CMEMS toolbox
```bash
pip install copernicusmarine
```

### 3. Download each dataset

```bash
# TWO-SAT SLA (for trend analysis — used in paper)
copernicusmarine subset \
  --dataset-id c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars \
  --variable sla --variable ugos --variable vgos \
  --start-datetime 1993-01-01 --end-datetime 2023-12-31 \
  --minimum-longitude -5 --maximum-longitude 10 \
  --minimum-latitude 35 --maximum-latitude 42

# ALL-SAT SLA (for eddy detection)
copernicusmarine subset \
  --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars \
  --variable sla --variable ugos --variable vgos \
  --start-datetime 1993-01-01 --end-datetime 2023-12-31 \
  --minimum-longitude -5 --maximum-longitude 10 \
  --minimum-latitude 35 --maximum-latitude 42

# MEDSEA Reanalysis Physics
copernicusmarine subset \
  --dataset-id cmems_mod_med_phy_my_4.2km_P1D-m \
  --variable uo --variable vo --variable thetao --variable zos --variable mlotst \
  --depth-min 0 --depth-max 1 \
  --start-datetime 1993-01-01 --end-datetime 2023-12-31

# SST L4 Observations
copernicusmarine subset \
  --dataset-id SST_MED_SST_L4_REP_OBSERVATIONS_010_021 \
  --variable analysed_sst \
  --start-datetime 1993-01-01 --end-datetime 2023-12-31

# BGC Plankton (Chl-a)
copernicusmarine subset \
  --dataset-id cmems_mod_med_bgc-plankton_my_4.2km_P1D-m_chl-phyc \
  --variable chl \
  --depth-min 0 --depth-max 1 \
  --start-datetime 1999-01-01 --end-datetime 2023-12-31
```

### 4. ERA5 MSLP (for WeMO reconstruction)
Register at: https://cds.climate.copernicus.eu (free)
Accept licence at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=download#manage-licences

Then run:
```bash
python 06_wemo_correlation/wemo_era5_reconstruction.py
```
(download is automatic via cdsapi)

## Expected File Sizes
| File | Size |
|---|---|
| TWO-SAT SLA | ~255 Mo |
| ALL-SAT SLA | ~1.8 Go |
| MEDSEA Reanalysis | ~5-10 Go |
| SST L4 | ~1.6 Go |
| BGC Chl-a | ~3.8 Go |
| ERA5 MSLP | ~5 Mo |

## After Download
Update the BASE path at the top of each script:
```python
BASE = r"C:\Users\YourName\YOUR_DATA_FOLDER"
```
