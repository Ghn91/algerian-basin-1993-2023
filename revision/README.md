# Revised analysis

```
revision_2026/
  *.py             scripts of the revised analysis (each one documents its
                   inputs, outputs and options in its header)
  donnees/         small derived inputs (monthly series, mean fields, masks,
                   ERA5 extracts)
  sorties/         outputs (tables in CSV, context in JSON);
                   sorties/eddy_atlas/ holds the regenerated eddy catalogue
  sorties_these/   copies of the thesis tables used by the figures
  manuscript/      figures and LaTeX tables of the revised manuscript
origine_2026-03/   outputs of the original analysis (March 2026): eddy
                   catalogue and reconstructed WeMO index, unchanged. The
                   March eddy catalogue is superseded: see "Eddy catalogue"
these/             chapter-3 scripts of the first author's doctoral thesis,
                   the modules they import, their derived inputs and tables
```

## Reproducing the figures

    cd revision/revision_2026
    python figures.py --toutes

Figures 1, 3, 4 and 6 to 9 run from the files present here. Figures 2 and 5,
and the scripts that start from the altimetry or reanalysis files, need the
NetCDF files larger than 50 MB: `download_altimetry.py` places the two
altimetry products in `revision_2026/donnees/` (run it without `ABASIN_DATA`
set), and the files of the original analysis are read from the directory
given by `ABASIN_DATA`. The thesis
scripts take the TWO-SAT file with `--netcdf`.

## Eddy catalogue

The eddy catalogue of the revised manuscript is
`revision_2026/sorties/eddy_atlas/`, produced by
`revision_2026/eddy_atlas.py` in the environment of
`environment_detection.yml`:

    conda env create -f environment_detection.yml
    conda activate eddy_detection
    cd revision/revision_2026
    python eddy_atlas.py --produire

Two properties of the software environment change the result.
py-eddy-tracker 3.6.1 depends on the contour routines of matplotlib, which
were modified in version 3.8: since then a whole contour level comes back as
a single path, and the library finds no more than a single eddy per polarity
and per daily map. The catalogue of version 1.0 (`origine_2026-03/`) was produced that way
and holds exactly one detection per date. And under Windows, numba's use of
the Intel short vector math library returns wrong eddy centres and areas;
`environment_detection.yml` disables it. `eddy_atlas.py` refuses to run with
matplotlib 3.8 or later and checks a synthetic field with known eddies before
producing anything. `eddy_detection_sensitivity.py` measures the sensitivity
of the catalogue to the filter cut-off and the contour interval.

Column names and some messages are in French.
