"""Paths and parameters of the thesis chain (read from params.yaml)."""

import os
import re
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parent

DATA_RAW = Path(os.environ.get("THESE_DATA", RACINE / "data" / "raw"))

DATA_INTERIM = RACINE / "data" / "interim"
DATA_PROC = RACINE / "data" / "processed"
FIG = RACINE / "figures"
TAB = RACINE / "tables"
DOCS = RACINE / "docs"

for _d in (DATA_RAW, DATA_INTERIM, DATA_PROC, FIG, TAB, DOCS):
    _d.mkdir(parents=True, exist_ok=True)


def params():
    with open(RACINE / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


DOSSIERS_HORS_AUDIT = ("_perimes", "_archive", "controles")


PROVENANCES_EXTERIEURES = {
    "RSMA-D-26-02810": "Algerian Basin altimetry, 1993-2023: values of the submitted article, imported, not recomputed by this chain (public code: github.com/Ghn91/algerian-basin-1993-2023, tag v1.0; doi:10.5281/zenodo.21420127)",
    "bassin-algerien-1993-2023": "Algerian Basin altimetry, 5W-10E / 35N-41N, 1993-2023: values recomputed by this chain on the domain and period of the article",
}


def provenance_declaree(libelle):
    return PROVENANCES_EXTERIEURES.get(libelle)


_MARGE_ACTIVE = None


def marge_active():
    return _MARGE_ACTIVE or params().get("analyse", {}).get("marge", "toutes")


SUFFIXES_DOMAINE = ("_bassin", "_nord")
_DOSSIER = re.compile(
    r"^(ch\d{2})_(?:fig|tab)\d+[a-z]?(" + "|".join(SUFFIXES_DOMAINE) + r")?(?:_|$)"
)


def dossier_de_sortie(identifiant):
    m = _DOSSIER.match(identifiant)
    return m.group(1) + (m.group(2) or "") if m else None


def etiquette_domaine():
    p = params()
    a = p.get("analyse", {})
    iso = a.get("isobathe") or p["domaine"]["isobathe_max"]
    marge = marge_active()
    k = p.get("clustering", {}).get("k_retenu")
    return f"{marge}-{int(iso)}m" + (f"-k{k}" if k else "")
