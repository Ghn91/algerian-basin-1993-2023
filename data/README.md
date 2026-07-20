# Data

The raw data are not stored here. Download them from the sources below and
point `config.py:DATA_DIR` at the folder that holds the NetCDF files.
Run `src/00_download_data.py` to fetch the CMEMS / C3S datasets
(free Copernicus account required).

- Altimetry (ALL-SAT, TWO-SAT), MEDSEA reanalysis, SST, biogeochemistry: Copernicus Marine Service — https://marine.copernicus.eu
- ERA5 mean sea-level pressure (WeMO reconstruction): Copernicus Climate Data Store (cdsapi)
- WeMO index (1993–2013): Climatology Group, University of Barcelona
- NAO index: NOAA Climate Prediction Center
