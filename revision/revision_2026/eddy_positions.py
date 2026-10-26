"""Distance of the eddy detections to the African coast and to the nearest
land, by polarity, the densest hexagon of each map, and the rank correlation
between mean chlorophyll and distance to land.

Usage: python eddy_positions.py --produire
"""

import csv
import io
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DOSSIER)
from zonation import distance_a_la_terre_km
from ekman_african_coast import composante_du_bord_sud
from zone_description import DERIVE, plus_proche
from figures import ATLAS, EMPRISE, LARGEUR_HEXAGONE_DEG, lire_atlas

SORTIE = os.path.join(DOSSIER, "sorties", "detection_positions.csv")


def distances_aux_points(dist_grille, lats, lons, plat, plon):
    ia, io_ = plus_proche(lats, plat), plus_proche(lons, plon)
    out = np.full(len(plat), np.nan)
    ok = (ia >= 0) & (io_ >= 0)
    out[ok] = dist_grille[ia[ok], io_[ok]]
    return out


def hexagone_maximal(lon, lat):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gx = int(round((EMPRISE[1] - EMPRISE[0]) / LARGEUR_HEXAGONE_DEG))
    fig, ax = plt.subplots()
    hb = ax.hexbin(lon, lat, gridsize=gx, extent=EMPRISE)
    c = hb.get_array()
    i = int(np.argmax(c))
    x, y = hb.get_offsets()[i]
    plt.close(fig)
    return float(x), float(y), int(c[i])


def resume(v):
    v = v[np.isfinite(v)]
    return {
        "n": int(v.size),
        "q25": float(np.percentile(v, 25)),
        "mediane": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
    }


def produire():
    z = np.load(DERIVE, allow_pickle=False)
    lats, lons = z["lats"], z["lons"]
    mer = z["etiquettes"] > 0
    afrique = composante_du_bord_sud(~mer, lats)
    d_afrique = np.full(mer.shape, np.nan)
    d_afrique[mer] = distance_a_la_terre_km(~afrique, lons, lats)[mer]
    d_terre = distance_a_la_terre_km(mer, lons, lats)
    lignes = []
    for k, p in ATLAS.items():
        a = lire_atlas(p)
        for nom, grille in (
            ("cote_africaine", d_afrique),
            ("terre_la_plus_proche", d_terre),
        ):
            r = resume(distances_aux_points(grille, lats, lons, a["lat"], a["lon"]))
            lignes.append(
                {
                    "polarite": k,
                    "distance_a": nom,
                    "n": r["n"],
                    "q25_km": f"{r['q25']:.4f}",
                    "mediane_km": f"{r['mediane']:.4f}",
                    "q75_km": f"{r['q75']:.4f}",
                }
            )
        h = hexagone_maximal(a["lon"], a["lat"])
        lignes.append(
            {
                "polarite": k,
                "distance_a": "hexagone_maximal",
                "n": h[2],
                "q25_km": "",
                "mediane_km": f"{h[0]:.2f}",
                "q75_km": f"{h[1]:.2f}",
            }
        )
    from scipy.stats import spearmanr

    chl = z["chl"].astype(float)
    ok = mer & np.isfinite(chl) & np.isfinite(d_terre)
    rho = float(spearmanr(d_terre[ok], chl[ok])[0])
    lignes.append(
        {
            "polarite": "chlorophylle",
            "distance_a": "rang_chl_distance_terre",
            "n": int(ok.sum()),
            "q25_km": "",
            "mediane_km": f"{rho:.6f}",
            "q75_km": "",
        }
    )
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    return 0


if __name__ == "__main__":
    if "--produire" in sys.argv:
        sys.exit(produire())
    print(__doc__)
