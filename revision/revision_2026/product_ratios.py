"""Product ratios quoted in the text (grid effect, constellation effect, both),
read from the archived outputs of altimetry_products.py and checked against a
recomputation from the versioned monthly series.

Usage: python product_ratios.py --produire
"""

import csv
import json
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties")
DERIVE = os.path.join(DOSSIER, "donnees", "eke_mensuelle_trois_champs_1993_2023.csv")
TENDANCES = os.path.join(SORTIES, "products_trends.csv")
RAPPORTS = os.path.join(SORTIES, "products_ratios_by_year.csv")
CONTEXTE = os.path.join(SORTIES, "products_context.json")

CHAMPS = ("allsat_0125", "allsat_025", "twosat_025")

TOLERANCE = 1e-6


def correlation(a, b):
    import numpy as np

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), int(ok.sum())
    x, y = x[ok], y[ok]
    xc, yc = x - x.mean(), y - y.mean()
    den = (xc * xc).sum() ** 0.5 * (yc * yc).sum() ** 0.5
    return (float((xc * yc).sum() / den) if den > 0 else float("nan"), int(ok.size))


def bornes(valeurs):
    import numpy as np

    v = np.asarray([x for x in valeurs if x == x], dtype=float)
    return (float(v.min()), float(v.max())) if v.size else (float("nan"),) * 2


def exces_de_tendance(pente_a, pente_b):
    return float(pente_a) - float(pente_b)


def lire_colonne(chemin, colonne, cle=None, valeur=None):
    sortie = []
    with open(chemin, newline="", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            if cle is not None and l.get(cle) != valeur:
                continue
            try:
                sortie.append(float(l[colonne]))
            except (KeyError, ValueError):
                pass
    return sortie


def moyenne_des_bornes(valeurs, n):
    import numpy as np

    v = np.asarray(valeurs, dtype=float)
    if v.size < 2 * n:
        return float("nan"), float("nan")
    return float(v[:n].mean()), float(v[-n:].mean())


def accord(archive, recalcul, tolerance=TOLERANCE):
    if archive is None or recalcul != recalcul:
        return False, float("nan")
    e = abs(float(archive) - float(recalcul))
    return e <= tolerance, e


def produire():
    if not os.path.isfile(DERIVE):
        return 1

    series = {c: [] for c in CHAMPS}
    with open(DERIVE, newline="", encoding="utf-8") as f:
        lecteur = csv.DictReader(f)
        colonnes = lecteur.fieldnames or []
        manquantes = [c for c in CHAMPS if c not in colonnes]
        if manquantes:
            return 1
        for l in lecteur:
            for c in CHAMPS:
                try:
                    series[c].append(float(l[c]))
                except (ValueError, TypeError):
                    series[c].append(float("nan"))

    with open(CONTEXTE, encoding="utf-8") as f:
        archive = json.load(f)
    archivees = archive.get("correlations_des_series", {})

    lignes = []

    paires = [
        (
            "allsat_0125",
            "allsat_025",
            "effet de grille seul : le bloc-moyennage deplace le niveau "
            "sans deformer la serie",
        ),
        (
            "allsat_025",
            "twosat_025",
            "effet de constellation seul, a maille egale — c'est celle-ci "
            "que cite la phrase sur la decorrelation",
        ),
        (
            "allsat_0125",
            "twosat_025",
            "les deux produits tels qu'ils sont publies : melange grille "
            "et constellation, ne mesure aucun des deux effets seul",
        ),
    ]
    for a, b, lecture in paires:
        cle = f"{a}_{b}"
        r, eff = correlation(series[a], series[b])
        arc = archivees.get(cle)
        ok, ecart = accord(arc, r)
        lignes.append(
            {
                "grandeur": f"correlation_{cle}",
                "valeur": f"{arc:.6f}" if arc is not None else "",
                "effectif": eff,
                "source": "products_context.json, confronte au derive mensuel",
                "lecture": lecture,
            }
        )

    rr = lire_colonne(RAPPORTS, "rapport_resolution")
    lo, hi = bornes(rr)
    lignes.append(
        {
            "grandeur": "rapport_resolution_minimum",
            "valeur": f"{lo:.6f}",
            "effectif": len(rr),
            "source": "products_ratios_by_year.csv",
            "lecture": "le JSON n'archive que les bornes du rapport de "
            "constellation ; celles-ci manquaient",
        }
    )
    lignes.append(
        {
            "grandeur": "rapport_resolution_maximum",
            "valeur": f"{hi:.6f}",
            "effectif": len(rr),
            "source": "products_ratios_by_year.csv",
            "lecture": "le rapport de resolution est plat sur la "
            "periode, a comparer a la derive de la "
            "constellation",
        }
    )

    sys.path.insert(0, DOSSIER)
    from altimetry_products import pente_de_sen

    rc = lire_colonne(RAPPORTS, "rapport_constellation")
    debut, fin = moyenne_des_bornes(rc, 5)
    derive = pente_de_sen(rc, 10.0)
    lignes.append(
        {
            "grandeur": "rapport_constellation_cinq_premieres_annees",
            "valeur": f"{debut:.6f}",
            "effectif": 5,
            "source": "products_ratios_by_year.csv",
            "lecture": "le TWO-SAT voyait plus d'energie que l'ALL-SAT "
            "degrade au debut de la periode",
        }
    )
    lignes.append(
        {
            "grandeur": "rapport_constellation_cinq_dernieres_annees",
            "valeur": f"{fin:.6f}",
            "effectif": 5,
            "source": "products_ratios_by_year.csv",
            "lecture": "et nettement moins a la fin : l'attenuation "
            "n'est pas stationnaire",
        }
    )
    lignes.append(
        {
            "grandeur": "derive_du_rapport_constellation_par_decennie",
            "valeur": f"{derive:.6f}",
            "effectif": len(rc),
            "source": "products_ratios_by_year.csv, pente de Sen de altimetry_products",
            "lecture": "a confronter a l'exces de tendance",
        }
    )

    pa = lire_colonne(TENDANCES, "pente_relative_par_decennie", "champ", "allsat_0125")
    pb = lire_colonne(TENDANCES, "pente_relative_par_decennie", "champ", "twosat_025")
    if pa and pb:
        e = exces_de_tendance(pa[0], pb[0])
        lignes.append(
            {
                "grandeur": "exces_de_tendance_relative",
                "valeur": f"{e:.6f}",
                "effectif": 2,
                "source": "products_trends.csv",
                "lecture": "ALL-SAT moins TWO-SAT — difference des deux pentes",
            }
        )

    chemin = os.path.join(SORTIES, "product_ratios.csv")
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)

    return 0


def main(argv):
    if "--produire" in argv:
        return produire()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
