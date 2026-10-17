"""Detection settings of the eddy catalogue as actually used: the call in
src/eddy_detection.py, the range of speed radii in the catalogue, the distance
of the detections to land, and the anticyclone/cyclone ratio by period.

Usage: python eddy_detection_parameters.py --relever
"""

import ast
import csv
import io
import json
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
SORTIES = os.path.join(DOSSIER, "sorties")
ATLAS = {
    "anticyclone": os.path.join(
        RACINE, "origine_2026-03", "anticyclones_1993_2023_ALL.csv"
    ),
    "cyclone": os.path.join(RACINE, "origine_2026-03", "cyclones_1993_2023_ALL.csv"),
}
MASQUE = os.path.join(DOSSIER, "donnees", "masque_cotier_medsea.npz")
SCRIPT_PUBLIC = os.path.join(
    os.path.dirname(RACINE), "algerian-basin-1993-2023", "src", "eddy_detection.py"
)

ARGS_ATTENDUS = 6


def arguments_de_l_appel(source, methode="eddy_identification"):
    try:
        arbre = ast.parse(source)
    except SyntaxError:
        return None
    for n in ast.walk(arbre):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == methode
        ):
            return len(n.args) + len(n.keywords)
    return None


def bornes(valeurs):
    v = [x for x in valeurs if x == x]
    return (min(v), max(v)) if v else (None, None)


PERIODES = (
    ("1993-2002", 1993, 2002),
    ("2003-2012", 2003, 2012),
    ("2013-2023", 2013, 2023),
    ("1re moitie 1993-2008", 1993, 2008),
    ("2e moitie 2009-2023", 2009, 2023),
    ("tout 1993-2023", 1993, 2023),
)


def comptes_par_periode(annees_a, annees_c, periodes=PERIODES):
    out = []
    for libelle, y0, y1 in periodes:
        na = sum(1 for y in annees_a if y0 <= y <= y1)
        nc = sum(1 for y in annees_c if y0 <= y <= y1)
        out.append((libelle, na, nc, (na / nc) if nc else None))
    return out


TEMOINS = (
    ("plein large", 2.0, 38.5, True),
    ("interieur du continent", 2.0, 35.1, False),
)


def sens_du_masque(mask, lons, lats, temoins=TEMOINS):
    import numpy as np

    def lu(lo, la):
        i = int(np.abs(np.asarray(lons) - lo).argmin())
        j = int(np.abs(np.asarray(lats) - la).argmin())
        return bool(mask[j, i])

    if all(lu(lo, la) == att for _, lo, la, att in temoins):
        return "mer"
    if all(lu(lo, la) != att for _, lo, la, att in temoins):
        return "terre"
    return None


def distance_minimale_a_la_terre(lon, lat, terre, lons, lats):
    import numpy as np

    jt, it = np.where(terre)
    if jt.size == 0:
        return float("inf")
    d = np.hypot(np.asarray(lons)[it] - lon, np.asarray(lats)[jt] - lat)
    return float(d.min())


def pas_de_grille(lons, lats):
    import numpy as np

    return float(min(np.diff(lons).min(), np.diff(lats).min()))


def relever():
    import numpy as np

    manquants = [c for c in list(ATLAS.values()) + [MASQUE] if not os.path.isfile(c)]
    if manquants:
        return 1

    n = None
    if os.path.isfile(SCRIPT_PUBLIC):
        n = arguments_de_l_appel(
            io.open(SCRIPT_PUBLIC, encoding="utf-8", errors="replace").read()
        )

    d = np.load(MASQUE)
    mask, lons, lats = d["mask"], d["lons"], d["lats"]
    pas = pas_de_grille(lons, lats)

    sens = sens_du_masque(mask, lons, lats)
    if sens is None:
        return 1
    terre = ~mask if sens == "mer" else mask

    lignes, annees, dates = [], {}, {}
    for pol, chemin in ATLAS.items():
        rs, re_, amp, dmin = [], [], [], []
        annees[pol], dates[pol] = [], set()
        with io.open(chemin, encoding="utf-8") as f:
            for x in csv.DictReader(f):
                annees[pol].append(int(x["date"][:4]))
                dates[pol].add(x["date"])
                rs.append(float(x["radius_s"]))
                re_.append(float(x["radius_e"]))
                amp.append(float(x["amplitude"]))
                dmin.append(
                    distance_minimale_a_la_terre(
                        float(x["lon"]), float(x["lat"]), terre, lons, lats
                    )
                )
        for nom, vals, fac, unite in (
            ("rayon de vitesse", rs, 1e-3, "km"),
            ("rayon effectif", re_, 1e-3, "km"),
            ("amplitude", amp, 1e2, "cm"),
            ("distance a la terre", dmin, 1.0, "deg"),
        ):
            lo, hi = bornes(vals)
            lignes.append(
                [pol, nom, unite, f"{lo * fac:.6f}", f"{hi * fac:.6f}", len(vals)]
            )
        pres = sum(1 for x in dmin if x <= pas * 1.001)
        lignes.append(
            [
                pol,
                "detections en premiere couronne",
                "nombre",
                str(pres),
                str(pres),
                len(dmin),
            ]
        )

    pola = comptes_par_periode(annees["anticyclone"], annees["cyclone"])
    un_par_jour = all(len(dates[p]) == len(annees[p]) for p in ATLAS)

    os.makedirs(SORTIES, exist_ok=True)
    cp = os.path.join(SORTIES, "detection_polarity_by_period.csv")
    with io.open(cp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "periode",
                "jours_anticycloniques",
                "jours_cycloniques",
                "rapport",
                "un_enregistrement_par_date",
            ]
        )
        for libelle, na, nc, r in pola:
            w.writerow(
                [
                    libelle,
                    na,
                    nc,
                    f"{r:.6f}" if r else "",
                    "oui" if un_par_jour else "non",
                ]
            )

    c = os.path.join(SORTIES, "detection_bounds.csv")
    with io.open(c, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["polarite", "grandeur", "unite", "minimum", "maximum", "n"])
        w.writerows(lignes)

    j = os.path.join(SORTIES, "detection_context.json")
    io.open(j, "w", encoding="utf-8", newline="\n").write(
        json.dumps(
            {
                "arguments_de_l_appel": n,
                "arguments_attendus": ARGS_ATTENDUS,
                "pas_de_grille_deg": pas,
                "appel_relu": os.path.isfile(SCRIPT_PUBLIC),
                "un_enregistrement_par_date": un_par_jour,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


def main(argv):
    if "--relever" in argv:
        return relever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
