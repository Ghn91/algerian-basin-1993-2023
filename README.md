# Thirty years of surface circulation variability in the Algerian Basin (1993–2023)

Reproducibility archive (code + workflow) for the study:

> **A. Nedjar, L. Nacef, F. Houma & N.E.I. Bachari** (in prep.).
> *Thirty years of surface circulation variability in the Algerian Basin
> (1993–2023): mesoscale energy trends, dominant modes, dynamic zonation,
> and eddy statistics from multi-source satellite and reanalysis records.*
> Submitted to **Regional Studies in Marine Science**.

Affiliations: LOBEM, FSB, USTHB (Algiers) · ECOSYSMarL, ENSSMAL (Algiers).

---

## What this repository does

A multi-method analysis of surface circulation in the Algerian Basin over the
full satellite-altimetry era, combining CMEMS/C3S altimetry with the MEDSEA
1/24° reanalysis:

- **Eddy kinetic energy (EKE)** trend with a bias-corrected two-satellite
  product — Mann–Kendall + Sen's slope (+2.06 cm² s⁻² decade⁻¹, 95 % CI
  +0.65 to +3.50) and a Morlet wavelet spectrum (dominant peak 74 months).
- **Pettitt change-point** test flagging a suggestive 2005–2008 transition.
- **EOF** modes of monthly SLA anomaly (leading mode 54.2 %).
- **Dynamic zonation** by Ward clustering of the mean velocity field
  (six regimes; Zone 1 = coastal upwelling).
- **Eddy catalogue** (py-eddy-tracker, 7 215 anticyclonic / 4 525 cyclonic).
- **WeMO / NAO coupling** — published UB index plus an ERA5 reconstruction
  extending the WeMO–EKE cross-correlation to 2023.

## Repository structure

```
.
├── config.py                 # edit DATA_DIR here (or set ABASIN_DATA env var)
├── requirements.txt
├── data/README.md            # dataset IDs + how to download (no data stored here)
├── src/
│   ├── 00_download_data.py           # fetch all CMEMS/C3S datasets
│   ├── fig02_mean_circulation.py     # Fig. 2  mean ADT / EKE / SLA std / MDT
│   ├── fig03_eke_timeseries.py       # Fig. 3  EKE time series (ALL-SAT)
│   ├── fig04_trend_wavelet.py        # Fig. 4  MK/Sen trend + wavelet (+ Sen 95% CI)
│   ├── fig05_eof.py                  # Fig. 5  EOF / PC modes
│   ├── fig06_zonation.py             # Fig. 6  Ward clustering + divergence
│   ├── fig07_eddy_atlas.py           # Fig. 7  eddy atlas
│   ├── fig08_wemo_era5_crosscorr.py  # Fig. 8  WeMO reconstruction + cross-corr
│   ├── eddy_detection.py             # py-eddy-tracker detection loop
│   ├── eddy_catalogue_stats.py       # catalogue assembly + Table 1
│   ├── analysis_pettitt.py           # Pettitt change-point test
│   └── analysis_wemo_nao_ub.py       # WeMO (UB) + NAO cross-correlation
├── extras/                    # diagnostics, animation (optional)
└── notebooks/                 # original exploratory notebooks (provenance)
```

## Quick start

```bash
# 1. environment (conda recommended for cartopy / py-eddy-tracker)
conda create -n abasin python=3.9
conda activate abasin
pip install -r requirements.txt

# 2. get the data (free Copernicus account required)
python src/00_download_data.py          # writes .nc files
#    then edit config.py -> DATA_DIR to point at that folder

# 3. reproduce a figure
python src/fig04_trend_wavelet.py        # also prints the Sen slope 95% CI
```

## Reproducing each result

| Output | Script |
|---|---|
| Fig. 2 mean circulation | `src/fig02_mean_circulation.py` |
| Fig. 3 EKE time series | `src/fig03_eke_timeseries.py` |
| Fig. 4 trend + wavelet, Sen 95% CI | `src/fig04_trend_wavelet.py` |
| Fig. 5 EOF modes | `src/fig05_eof.py` |
| Fig. 6 dynamic zonation | `src/fig06_zonation.py` |
| Fig. 7 eddy atlas | `src/fig07_eddy_atlas.py` |
| Fig. 8 WeMO–EKE | `src/fig08_wemo_era5_crosscorr.py` |
| Table 1 eddy statistics | `src/eddy_catalogue_stats.py` |
| Pettitt change point | `src/analysis_pettitt.py` |
| Eddy detection (long run) | `src/eddy_detection.py` |

## Data availability

No raw data are committed (see `.gitignore`). Every dataset is public — dataset
IDs and download commands are in [`data/README.md`](data/README.md). Altimetry
and reanalysis are from CMEMS/C3S; ERA5 MSLP from the CDS; WeMO from the
University of Barcelona; NAO from NOAA CPC.

## Citation

```bibtex
@article{nedjar_algerian_basin_2026,
  author  = {Nedjar, A. and Nacef, L. and Houma, F. and Bachari, N.E.I.},
  title   = {Thirty years of surface circulation variability in the
             Algerian Basin (1993--2023)},
  journal = {Regional Studies in Marine Science},
  year    = {2026},
  note    = {in preparation}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contact

Abdelghani Nedjar — aneddjar@usthb.dz · abdelghani.nedjar91@gmail.com
