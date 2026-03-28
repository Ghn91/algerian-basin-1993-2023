# Algerian Basin Surface Circulation 1993–2023
## Analysis Code — Ocean Science Submission

**Title:** Surface circulation variability of the Algerian Basin (1993–2023): mesoscale energy intensification, dynamical zonation, and atmospheric forcing

**Authors:** A. Nedjar¹, N.E.I. Bachari¹, F. Houma²

¹ LOBEM, Faculté des Sciences Biologiques, USTHB, Bab Ezzouar, Alger, Algeria  
² ECOSYSMarL, ENSSMAL, Dély Ibrahim, Alger, Algeria

**ORCID (corresponding author):** [0009-0003-0676-0772](https://orcid.org/0009-0003-0676-0772)

**Target journal:** Ocean Science (EGU / Copernicus)

---

## Repository Structure

```
algerian-basin-1993-2023/
│
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── environment.yml                  ← conda environment
│
├── 01_data_preprocessing/
│   ├── diagnostic_cmems.py          ← NetCDF file inspection
│   └── prepare_monthly_fields.py    ← Resampling CMEMS daily → monthly
│
├── 02_eke_trend_analysis/
│   ├── eke_mannkendall_tfpw.py      ← EKE trend: Mann-Kendall + TFPW
│   ├── eke_pettitt_test.py          ← Change-point detection (2005)
│   └── eke_seasonal_cycle.py        ← Seasonal decomposition
│
├── 03_wavelet_eof/
│   ├── morlet_wavelet_eke.py        ← Morlet wavelet scalogram (74-month peak)
│   └── eof_sla_decomposition.py     ← EOF decomposition (EOF1 = 54.2%)
│
├── 04_eddy_detection/
│   ├── eddy_detection_pyeddytracker.py  ← py-eddy-tracker v3.6.1 wrapper
│   ├── eddy_catalogue_stats.py          ← AC/CY statistics and annual counts
│   └── eddy_spatial_density.py          ← Spatial density maps
│
├── 05_cah_zonation/
│   ├── ward_hierarchical_clustering.py  ← Ward/CAH N=6 (CH=11,056)
│   └── divergence_zone1.py              ← Horizontal divergence Zone 1
│
├── 06_wemo_correlation/
│   ├── wemo_era5_reconstruction.py      ← WeMO index from ERA5 MSLP
│   ├── wemo_cross_correlation.py        ← Cross-correlation WeMO–EKE 1993-2023
│   └── wemo_validation_1993_2013.py     ← Validation vs published record
│
├── 07_upwelling_validation/
│   └── upwelling_zone1_sst_chl.py       ← SST + Chl-a validation Zone 1
│
├── 08_figures/
│   ├── figure_01_mean_circulation.py
│   ├── figure_02_eke_timeseries.py
│   ├── figure_03_wavelet_trend.py
│   ├── figure_04_eof_sla.py
│   ├── figure_05_cah_zonation.py
│   ├── figure_06_eddy_atlas.py
│   └── figure_07_wemo_eke.py
│
└── data/
    └── README_data.md               ← Data access instructions (no raw data here)
```

---

## Data Access

All input data are from the Copernicus Marine Service (CMEMS) and are freely available:

| Dataset | Product ID | Period |
|---|---|---|
| SLA TWO-SAT (DUACS DT2021) | `C3S-SLEV-L4-DUACS-TWOSAT-0.25DEG` | 1993–2023 |
| SLA ALL-SAT (DUACS DT2021) | `CMEMS-SL-GLO-PHY-ALLSAT-0.125DEG` | 1993–2023 |
| MEDSEA Reanalysis (currents, temp, SSH, MLD) | `MEDSEA_MULTIYEAR_PHY_006_004` | 1993–2023 |
| SST L4 Observations | `SST_MED_SST_L4_REP_OBSERVATIONS_010_021` | 1993–2023 |
| BGC Plankton (Chl-a) | `MEDSEA_MULTIYEAR_BGC_006_008` | 1999–2023 |
| ERA5 MSLP (WeMO reconstruction) | `reanalysis-era5-single-levels-monthly-means` | 1993–2023 |

Download instructions: [CMEMS Marine Data Store](https://marine.copernicus.eu) and [Copernicus Climate Data Store](https://cds.climate.copernicus.eu)

---

## Key Results

| Result | Value |
|---|---|
| EKE trend (TWO-SAT) | +2.06 cm²/s²/decade (τ=0.099, p=0.004) |
| EKE trend (ALL-SAT, biased) | +15.99 cm²/s²/decade |
| Pettitt change point | 2005 (U=118.0, p=0.068) |
| Dominant wavelet period | 74 months (~6.2 years) |
| WeMO–EKE cross-correlation | r=−0.220, p=0.00002, n=372 |
| WeMO lead time | ~1–2 months |
| NAO–EKE correlation | r=+0.038, p=0.462 (NS) |
| EOF1 variance | 54.2% |
| CAH optimal N | 6 (Calinski-Harabasz = 11,056) |
| Zone 1 divergence | +1.18×10⁻⁸ s⁻¹ (w ≈ 0.05 m/day) |
| Anticyclones detected | 7,215 daily observations |
| Cyclones detected | 4,525 daily observations |

---

## Installation

```bash
git clone https://github.com/Ghn91/algerian-basin-1993-2023.git
cd algerian-basin-1993-2023
conda env create -f environment.yml
conda activate eddy_env
```

Or with pip:
```bash
pip install -r requirements.txt
```

---

## Dependencies

- Python 3.10+
- xarray, numpy, scipy, matplotlib
- cartopy, cmocean
- py-eddy-tracker v3.6.1
- statsmodels (Mann-Kendall, TFPW)
- scikit-learn (Ward clustering)
- cdsapi (ERA5 download)

See `requirements.txt` and `environment.yml` for exact versions.

---

## Usage

Run analyses in order:

```bash
# 1. Inspect your CMEMS files
python 01_data_preprocessing/diagnostic_cmems.py

# 2. EKE trend analysis
python 02_eke_trend_analysis/eke_mannkendall_tfpw.py

# 3. Wavelet + EOF
python 03_wavelet_eof/morlet_wavelet_eke.py
python 03_wavelet_eof/eof_sla_decomposition.py

# 4. Eddy detection (requires py-eddy-tracker)
python 04_eddy_detection/eddy_detection_pyeddytracker.py

# 5. CAH zonation
python 05_cah_zonation/ward_hierarchical_clustering.py

# 6. WeMO analysis (requires ERA5 file)
python 06_wemo_correlation/wemo_era5_reconstruction.py
python 06_wemo_correlation/wemo_cross_correlation.py

# 7. Upwelling validation
python 07_upwelling_validation/upwelling_zone1_sst_chl.py
```

---

## Citation

If you use this code, please cite:

> Nedjar, A., Bachari, N.E.I., and Houma, F.: Surface circulation variability of the Algerian Basin (1993–2023): mesoscale energy intensification, dynamical zonation, and atmospheric forcing, *Ocean Science*, under review, 2026.

---

## License

MIT License — see `LICENSE` file.

## Contact

Corresponding author: Abdelghani Nedjar  
ORCID: [0009-0003-0676-0772](https://orcid.org/0009-0003-0676-0772)  
Email: contact via GitHub profile
