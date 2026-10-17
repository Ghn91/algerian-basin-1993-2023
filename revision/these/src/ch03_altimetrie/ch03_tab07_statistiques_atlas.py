"""Statistics of the eddy catalogue by polarity (amplitude, radii, speed,
fraction of documented days).

Usage: python ch03_tab07_statistiques_atlas.py [--source DIR]
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))
import pandas as pd
from provenance import sauver_table

SOURCE_DEFAUT = RACINE / "article" / "origine_2026-03"
ARCHIVE = RACINE / "docs" / "ch03_atlas.json"

FICHIERS = {
    "anticyclone": "anticyclones_1993_2023_ALL.csv",
    "cyclone": "cyclones_1993_2023_ALL.csv",
}

PREMIER_JOUR, DERNIER_JOUR = "1993-01-01", "2023-12-31"

COLONNES = {
    "amplitude": ("cm", 100.0),
    "radius_s": ("km", 1.0 / 1000.0),
    "radius_e": ("km", 1.0 / 1000.0),
    "speed_average": ("cm_par_s", 100.0),
    "shape_error_s": ("pourcent", 1.0),
}
AIRES_ECARTEES = ("speed_area", "effective_area")


def jours_de_calendrier(premier=PREMIER_JOUR, dernier=DERNIER_JOUR):
    from datetime import date

    a = date(*(int(x) for x in premier.split("-")))
    b = date(*(int(x) for x in dernier.split("-")))
    return (b - a).days + 1


def lire(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        entetes = lecteur.fieldnames or []
        lignes = [
            {k: v for k, v in r.items() if k not in AIRES_ECARTEES} for r in lecteur
        ]
    return lignes, entetes


def statistiques(lignes):
    if not lignes:
        raise SystemExit("empty atlas: nothing to summarise")
    cadre = pd.DataFrame(lignes)
    sortie = {
        "detections": len(lignes),
        "dates_distinctes": cadre["date"].nunique(),
        "premiere_date": cadre["date"].min(),
        "derniere_date": cadre["date"].max(),
    }
    for colonne, (unite, facteur) in COLONNES.items():
        valeurs = pd.to_numeric(cadre[colonne], errors="coerce") * facteur
        sortie[f"{colonne}_moyenne_{unite}"] = float(valeurs.mean())
        sortie[f"{colonne}_ecart_type_{unite}"] = float(valeurs.std())
        sortie[f"{colonne}_valeurs_manquantes"] = int(valeurs.isna().sum())
    return sortie


def empreinte(chemin, bloc=1 << 20):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(bloc), b""):
            h.update(morceau)
    return h.hexdigest()


def produire(dossier_source):
    source = Path(dossier_source)
    manquants = [n for n in FICHIERS.values() if not (source / n).exists()]
    if manquants:
        raise SystemExit(
            f"source not found under {source}: {', '.join(manquants)}\nGive the atlas directory with --source."
        )

    jours = jours_de_calendrier()

    lignes_table, archive = [], {}
    for signe, nom in FICHIERS.items():
        chemin = source / nom
        lignes, entetes = lire(chemin)
        s = statistiques(lignes)
        ecartees = [c for c in AIRES_ECARTEES if c in entetes]
        lignes_table.append(
            {
                "signe": signe,
                "detections": s["detections"],
                "dates_distinctes": s["dates_distinctes"],
                "jours_de_calendrier": jours,
                "part_des_jours_documentes": f"{s['detections'] / jours:.4f}",
                "amplitude_moyenne_cm": f"{s['amplitude_moyenne_cm']:.2f}",
                "amplitude_ecart_type_cm": f"{s['amplitude_ecart_type_cm']:.2f}",
                "rayon_vitesse_moyen_km": f"{s['radius_s_moyenne_km']:.1f}",
                "rayon_vitesse_ecart_type_km": f"{s['radius_s_ecart_type_km']:.1f}",
                "rayon_effectif_moyen_km": f"{s['radius_e_moyenne_km']:.1f}",
                "rayon_effectif_ecart_type_km": f"{s['radius_e_ecart_type_km']:.1f}",
                "vitesse_moyenne_cm_par_s": f"{s['speed_average_moyenne_cm_par_s']:.2f}",
                "vitesse_ecart_type_cm_par_s": f"{s['speed_average_ecart_type_cm_par_s']:.2f}",
                "erreur_de_forme_moyenne_pourcent": f"{s['shape_error_s_moyenne_pourcent']:.2f}",
                "valeurs_manquantes": sum(
                    s[f"{c}_valeurs_manquantes"] for c in COLONNES
                ),
            }
        )
        archive[signe] = {
            "fichier": nom,
            "sha256": empreinte(chemin),
            "octets": chemin.stat().st_size,
            "premiere_date": s["premiere_date"],
            "derniere_date": s["derniere_date"],
            "colonnes_ecartees": ecartees,
            "detections": s["detections"],
            "dates_distinctes": s["dates_distinctes"],
            "une_seule_par_jour": s["detections"] == s["dates_distinctes"],
        }

    sauver_table(
        pd.DataFrame(lignes_table),
        "ch03_tab07",
        "statistiques de l atlas de tourbillons",
        "src/ch03_altimetrie/ch03_tab07_statistiques_atlas.py",
        " · ".join(FICHIERS.values()),
        "3.9",
        domaine="bassin-algerien-1993-2023",
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "source": source.name,
                "calendrier": {
                    "premier": PREMIER_JOUR,
                    "dernier": DERNIER_JOUR,
                    "jours": jours,
                },
                "convention_ecart_type": "diviseur n - 1, comme le script d'origine",
                "fichiers": archive,
                "note": "les colonnes d'aire sont ecartees a la LECTURE ; le compte "
                "d'erreurs du script d'origine n'a jamais ete archive, donc un "
                "jour sans detection peut etre une absence ou un echec",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes_table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=str(SOURCE_DEFAUT))
    a = ap.parse_args()
    produire(a.source)
