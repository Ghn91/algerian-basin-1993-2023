# Data

None of the raw data are stored in this repository (files are large and
publicly hosted). All datasets are freely available from the Copernicus
Marine Service (CMEMS), the Copernicus Climate Change Service (C3S) and
ERA5 (CDS). Re-download them with `src/00_download_data.py`
(requires a free Copernicus account and the `copernicusmarine` toolbox).

## CMEMS / C3S datasets used

| dataset_id | variables |
|---|---|
| `cmems_mod_med_phy-temp_my_4.2km_P1D-m` | ["thetao"] |
| `cmems_mod_med_phy-sal_my_4.2km_P1D-m` | ["so"] |
| `cmems_mod_med_phy-ssh_my_4.2km_P1D-m` | ["zos"] |
| `cmems_mod_med_phy-mld_my_4.2km_P1D-m` | ["mlotst"] |
| `cmems_mod_glo_phy_my_0.083deg_P1D-m` | ["thetao", "uo", "vo", "so", "zos", "mlotst"] |
| `cmems_mod_med_bgc-bio_my_4.2km_P1D-m` | ["o2"] |
| `cmems_mod_med_bgc-nut_my_4.2km_P1D-m` | ["no3", "po4"] |
| `cmems_mod_med_bgc-plankton_my_4.2km_P1D-m` | ["chl", "phyc"] |
| `cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021` | ["analysed_sst", "analysis_error"] |
| `cmems_obs-ins_med_phybgcwav_mynrt_na_irr` | ["TEMP", "PSAL", "DOXY"] |

## Additional indices
- **WeMO (instrumental, 1993-2013)** - Climatology Group, University of Barcelona: http://www.ub.es/gc/
- **WeMO (ERA5 reconstruction, 1993-2023)** - built in `src/fig08_wemo_era5_crosscorr.py` from ERA5 MSLP (CDS, `cdsapi`), San Fernando - Padua station pair.
- **NAO (monthly)** - NOAA Climate Prediction Center.

After download, point `config.py:DATA_DIR` at the folder holding the `.nc` files.
