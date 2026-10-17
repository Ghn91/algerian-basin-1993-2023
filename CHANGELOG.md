# Changelog

## v1.1 (revised manuscript)

### Corrections to the original scripts

- `src/fig08_wemo_era5_crosscorr.py`: the WeMO index is computed with its canonical sign (San Fernando minus Padua); v1.0 computed the opposite.
- `src/fig08_wemo_era5_crosscorr.py`: the lag axis label states the convention the computation uses (a positive lag means that the WeMO leads).
- `src/fig08_wemo_era5_crosscorr.py`: the generated paragraph reports the signed lag instead of its absolute value.
- `src/fig08_wemo_era5_crosscorr.py`: same, second occurrence.
- `src/fig08_wemo_era5_crosscorr.py`: user-independent path in a docstring.
- `src/analysis_pettitt.py`: the generated sentence states significance only when p < 0.05.
- `src/fig03_eke_timeseries.py`: unused import of pymannkendall removed (the trend line of the figure is a least-squares fit).
- `config.py`: data directory taken from ABASIN_DATA (default: ./data).

The Figure 8 script of the original analysis correlates the index with the
kinetic energy of the total geostrophic flow, computed from monthly-mean
velocities. It is left as it was, so that the submitted figure remains
reproducible; the revised analysis computes both this quantity and the EKE of
Eq. 1 (`revision/these/src/ch03_altimetrie/ch03_tab20_correlation_wemo.py`).

### Added

- `revision/`: scripts, derived inputs, outputs and thesis tables of the
  revised analysis, and the outputs of the original analysis
  (`revision/origine_2026-03/`).
- `MANIFEST.csv`: every file, its SHA-256 and its origin.

### Removed

- `GITIGN~1` and `REQUIR~1.TXT`, truncated copies of `.gitignore` and
  `requirements.txt`, and the compiled `__pycache__` files.
- The notebooks, which remain in version 1.0.
