# Surface circulation variability in the Algerian Basin (1993–2023)

Analysis code for:

Nedjar A., Nacef L., Houma F., Bachari N.E.I. Thirty years of surface
circulation variability in the Algerian Basin (1993–2023): mesoscale
energy trends, dominant modes, dynamic zonation, and eddy statistics
from multi-source satellite and reanalysis records. Regional Studies in
Marine Science.

## Data

All datasets are publicly available (CMEMS, C3S, ERA5, NOAA CPC). Dataset
identifiers and the download commands are listed in `data/README.md`.
After downloading, set the folder in `config.py` (`DATA_DIR`).

## Environment

    conda env create -f environment.yml     # or: pip install -r requirements.txt

## Scripts

`src/`

    00_download_data.py        download the CMEMS / C3S datasets
    fig02_mean_circulation.py  Figure 2
    fig03_eke_timeseries.py    Figure 3
    fig04_trend_wavelet.py     Figure 4  (Mann-Kendall / Sen + wavelet)
    fig05_eof.py               Figure 5  (EOF / PC modes)
    fig06_zonation.py          Figure 6  (Ward clustering + divergence)
    fig07_eddy_atlas.py        Figure 7  (eddy atlas)
    fig08_wemo_era5_crosscorr.py  Figure 8  (WeMO–EKE cross-correlation)
    eddy_detection.py          eddy detection (py-eddy-tracker)
    eddy_catalogue_stats.py    eddy catalogue + Table 1
    analysis_pettitt.py        Pettitt change-point test
    analysis_wemo_nao_ub.py    WeMO / NAO cross-correlation

The study-area map (Figure 1) is produced with a separate script.
