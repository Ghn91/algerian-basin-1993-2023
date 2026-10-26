"""Sign of the WeMO reconstructions against the published index.

Usage: python ch03_tab18_convention_wemo.py [--source-article PATH] [--source-ub PATH]
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))

import pandas as pd
from provenance import sauver_table

SOURCE_UB = Path("src/analysis_wemo_nao_ub.py")
SOURCE_ARTICLE = Path("revision/origine_2026-03/wemo_index_era5_1993_2023.csv")
TABLE_THESE = RACINE / "tables" / "ch05" / "ch05_tab23_indices_reconstruits_era5.csv"

ARCHIVE = RACINE / "docs" / "ch03_convention_wemo.json"
RISQUE = 0.01
Z_RISQUE = 2.5758293035489004


def seuil_indetermination(n, z=Z_RISQUE):
    n = int(n)
    if n < 4:
        return 1.0
    return float(np.tanh(z / np.sqrt(n - 3)))


def convention(serie, arbitre, z=Z_RISQUE):
    a = np.asarray(serie, dtype=float)
    b = np.asarray(arbitre, dtype=float)
    n = len(a)
    if n != len(b):
        raise SystemExit("the two series do not have the same length")
    if n < 4 or a.std() == 0 or b.std() == 0:
        return "indetermine", float("nan"), n, 1.0
    r = float(np.corrcoef(a, b)[0, 1])
    seuil = seuil_indetermination(n, z)
    if abs(r) < seuil:
        return "indetermine", r, n, seuil
    return ("conforme" if r > 0 else "inversee"), r, n, seuil


def lire_serie_ub(chemin):
    texte = Path(chemin).read_text(encoding="utf-8", errors="replace")
    bloc = re.search(r"wemo_data\s*=\s*\{(.*?)\n\}", texte, re.S)
    if bloc is None:
        raise SystemExit(
            f"no `wemo_data` block in {chemin}.\nGive another path with --source-ub."
        )
    serie = {}
    for an, valeurs in re.findall(r"(\d{4})\s*:\s*\[([^\]]*)\]", bloc.group(1)):
        mois = [v.strip() for v in valeurs.split(",") if v.strip()]
        if len(mois) != 12:
            raise SystemExit(f"year {an} carries {len(mois)} values instead of twelve.")
        for m, v in enumerate(mois, 1):
            serie[f"{int(an):04d}-{m:02d}"] = v
    if not serie:
        raise SystemExit("the `wemo_data` block is empty.")
    return serie


def lire_serie_article(chemin):
    import csv

    with open(chemin, newline="", encoding="utf-8") as f:
        return {r["time"]: r["wemo_era5"] for r in csv.DictReader(f)}


def lire_serie_these(chemin):
    import csv

    with open(chemin, newline="", encoding="utf-8") as f:
        return {
            f'{int(r["annee"]):04d}-{int(r["mois"]):02d}': r["wemo_era5"]
            for r in csv.DictReader(f)
        }


def empreinte(chemin, bloc=1 << 20):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(bloc), b""):
            h.update(morceau)
    return h.hexdigest()


def alignees(*series):
    communs = sorted(set.intersection(*(set(s) for s in series)))
    return communs, [np.array([float(s[m]) for m in communs]) for s in series]


def produire(
    source_ub=SOURCE_UB, source_article=SOURCE_ARTICLE, table_these=TABLE_THESE
):
    for chemin, quoi in (
        (source_ub, "le script qui porte la serie publiee"),
        (source_article, "la reconstruction de l'article"),
        (table_these, "la reconstruction de la these"),
    ):
        if not Path(chemin).exists():
            raise SystemExit(f"source not found: {chemin}\n({quoi})")

    ub = lire_serie_ub(source_ub)
    art = lire_serie_article(source_article)
    the = lire_serie_these(table_these)

    sauver_table(
        pd.DataFrame([{"mois": m, "wemo_ub_publiee": ub[m]} for m in sorted(ub)]),
        "ch03_tab18",
        "serie wemo publiee ub barcelone transportee",
        "src/ch03_altimetrie/ch03_tab18_convention_wemo.py",
        Path(source_ub).name,
        "3.7",
        domaine="RSMA-D-26-02810",
    )

    lignes = []
    for nom, serie in (("article", art), ("these", the)):
        mois, (a, b) = alignees(serie, ub)
        verdict, r, n, seuil = convention(a, b)
        lignes.append(
            {
                "reconstruction": nom,
                "verdict": verdict,
                "r_contre_arbitre": r,
                "valeur_absolue": abs(r),
                "n_mois": n,
                "seuil_indetermination": seuil,
                "premier_mois": mois[0],
                "dernier_mois": mois[-1],
            }
        )

    mois3, (A, T, U) = alignees(art, the, ub)
    r_art, r_the = abs(np.corrcoef(A, U)[0, 1]), abs(np.corrcoef(T, U)[0, 1])
    r_entre = float(np.corrcoef(A, T)[0, 1])

    lignes.append(
        {
            "reconstruction": "ecart de qualite",
            "verdict": "article meilleur" if r_art > r_the else "these meilleure",
            "r_contre_arbitre": r_art - r_the,
            "valeur_absolue": abs(r_art - r_the),
            "n_mois": len(mois3),
            "seuil_indetermination": seuil_indetermination(len(mois3)),
            "premier_mois": mois3[0],
            "dernier_mois": mois3[-1],
        }
    )

    sauver_table(
        pd.DataFrame(
            [
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in L.items()}
                for L in lignes
            ]
        ),
        "ch03_tab19",
        "convention de signe des reconstructions wemo",
        "src/ch03_altimetrie/ch03_tab18_convention_wemo.py",
        Path(source_ub).name,
        "3.7",
        domaine="bassin-algerien-1993-2023",
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "arbitre": {
                    "fichier": str(Path(source_ub).name),
                    "sha256": empreinte(source_ub),
                    "mois": len(ub),
                    "premier": min(ub),
                    "dernier": max(ub),
                    "reserve": "serie saisie en dur dans un script, non telechargee de "
                    "la source ; sa fidelite a la publication de l'UB n'est "
                    "pas etablie ici. Ce qui est etabli porte sur les "
                    "conventions relatives des trois series.",
                },
                "article": {
                    "fichier": str(Path(source_article).name),
                    "sha256": empreinte(source_article),
                    "mois": len(art),
                },
                "these": {
                    "fichier": str(Path(table_these).name),
                    "sha256": empreinte(table_these),
                    "mois": len(the),
                },
                "mois_communs_aux_trois": len(mois3),
                "risque_du_seuil": RISQUE,
                "r_article_contre_these": r_entre,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-ub", default=str(SOURCE_UB))
    ap.add_argument("--source-article", default=str(SOURCE_ARTICLE))
    a = ap.parse_args()
    produire(a.source_ub, a.source_article)
