# Revised analysis

```
revision_2026/
  *.py             scripts of the revised analysis (each one documents its
                   inputs, outputs and options in its header)
  donnees/         small derived inputs (monthly series, mean fields, masks,
                   ERA5 extracts)
  sorties/         outputs (tables in CSV, context in JSON)
  sorties_these/   copies of the thesis tables used by the figures
  manuscript/      figures and LaTeX tables of the revised manuscript
origine_2026-03/   outputs of the original analysis (March 2026): eddy
                   catalogue and reconstructed WeMO index, unchanged
these/             chapter-3 scripts of the first author's doctoral thesis,
                   the modules they import, their derived inputs and tables
```

## Reproducing the figures

    cd revision/revision_2026
    python figures.py --toutes

Figures 1 and 3 to 9 run from the files present here. Figure 2 and the
scripts that start from the altimetry or reanalysis files need the NetCDF
files larger than 50 MB (downloaded by `download_altimetry.py` and the
scripts of `src/`); set `ABASIN_DATA` to their directory. The thesis scripts
take the TWO-SAT file with `--netcdf`.

Column names and some messages are in French.
