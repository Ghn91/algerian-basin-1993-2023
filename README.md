# Thirty years of surface circulation variability in the Algerian Basin (1993–2023)

Code, derived data and outputs of the study by **A. Nedjar, L. Nacef, F. Houma
and N.E.I. Bachari** (LOBEM, FSB, USTHB, Algiers; ECOSYSMarL, ENSSMAL, Algiers),
submitted to *Regional Studies in Marine Science*.

**Version 1.1** accompanies the revised manuscript. Version 1.0 accompanied the
submitted manuscript and remains available from the same Zenodo record
(doi:10.5281/zenodo.21420127); the differences are listed in
[CHANGELOG.md](CHANGELOG.md).

## Contents

```
src/                 scripts of the original analysis (v1.0), with the
                     corrections listed in CHANGELOG.md
extras/              diagnostics and animation of the original analysis
revision/            scripts, derived inputs and outputs of the revised
                     analysis; see revision/README.md
MANIFEST.csv         every file of this version, its size, SHA-256 and origin
```

The scripts of `src/` reproduce the figures of the submitted manuscript. The
figure captions and the sentences some of them print are those of the
submitted version; the results of the revised manuscript are produced by the
scripts of `revision/`. The exploratory notebooks of the original analysis
are part of version 1.0.

## Data

Files larger than 50 MB are not stored. All datasets are public: Copernicus
Marine Service, Copernicus Climate Change Service, ERA5 (Copernicus Climate
Data Store), the NOAA Climate Prediction Center and the Climatic Research Unit
(University of East Anglia). Download scripts: `src/00_download_data.py`,
`revision/revision_2026/download_altimetry.py` and
`revision/revision_2026/download_era5.py`. Scripts that need the large files
read their location from the environment variable `ABASIN_DATA` or from a
command-line option.

## Licence

MIT; see [LICENSE](LICENSE).

## Contact

Abdelghani Nedjar, abdelghani.nedjar91@gmail.com
