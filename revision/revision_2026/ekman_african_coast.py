"""Offshore Ekman transport along the African coast only (the land mass that
touches the southern edge of the domain), by season, coastal distance and
one-degree longitude band, with the contiguous bands of each sign.

Usage: python ekman_african_coast.py --produire
"""

import csv
import json
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")

sys.path.insert(0, DOSSIER)
from ekman_index import (
    RAYONS_KM,
    RAYON_DE_REFERENCE,
    geometrie_cotiere,
    indice_vers_le_large,
    interpoler_serie,
    masque_cotier,
)

TENSION = os.path.join(DONNEES, "era5_tension_mensuelle_1993_2023.nc")

MOIS_ETE = (6, 7, 8)
MOIS_HIVER = (12, 1, 2)
LARGEUR_BANDE = 1.0

TEMOIN_AFRIQUE = (35.3, 3.0)
TEMOIN_HORS_AFRIQUE = (40.5, 0.0)


def composante_du_bord_sud(terre, lats=None):
    import numpy as np
    from scipy import ndimage

    if lats is not None:
        la = np.asarray(lats, dtype=float)
        if la.size > 1 and la[1] <= la[0]:
            raise ValueError(
                "composante_du_bord_sud: latitude must increase with the row index"
            )
    etiquettes, _ = ndimage.label(np.asarray(terre, dtype=bool))
    au_sud = set(np.unique(etiquettes[0, :])) - {0}
    if not au_sud:
        return np.zeros_like(etiquettes, dtype=bool)
    return np.isin(etiquettes, sorted(au_sud))


def points_africains(mer, lons, lats, lon_cote, lat_cote):
    import numpy as np
    from scipy import ndimage

    afrique = composante_du_bord_sud(~np.asarray(mer, dtype=bool), lats)
    voisinage = ndimage.binary_dilation(afrique, structure=np.ones((3, 3)))
    la = np.asarray(lats, dtype=float)
    lo = np.asarray(lons, dtype=float)
    i = np.abs(la[:, None] - np.asarray(lat_cote)[None, :]).argmin(axis=0)
    j = np.abs(lo[:, None] - np.asarray(lon_cote)[None, :]).argmin(axis=0)
    return voisinage[i, j]


def dans_la_composante(masque, lons, lats, lat_pt, lon_pt):
    import numpy as np

    i = int(np.abs(np.asarray(lats, dtype=float) - lat_pt).argmin())
    j = int(np.abs(np.asarray(lons, dtype=float) - lon_pt).argmin())
    return bool(np.asarray(masque, dtype=bool)[i, j])


def bandes_de_longitude(lon_cote, largeur=LARGEUR_BANDE):
    import numpy as np

    lo = np.asarray(lon_cote, dtype=float)
    debut = float(np.floor(lo.min() / largeur) * largeur)
    fin = float(np.ceil(lo.max() / largeur) * largeur)
    bornes = []
    b = debut
    while b < fin - 1e-9:
        bornes.append((b, b + largeur))
        b += largeur
    return bornes


def diagnostic_des_bandes(lon_cote, lat_cote, bandes):
    import numpy as np

    lo = np.asarray(lon_cote, dtype=float)
    la = np.asarray(lat_cote, dtype=float)
    lignes = []
    for a, b in bandes:
        dans = (lo >= a) & (lo < b)
        n = int(dans.sum())
        lignes.append(
            {
                "lon_min": a,
                "lon_max": b,
                "n_points": n,
                "lat_min": float(la[dans].min()) if n else float("nan"),
                "lat_max": float(la[dans].max()) if n else float("nan"),
                "etendue_lat": (
                    float(la[dans].max() - la[dans].min()) if n else float("nan")
                ),
            }
        )
    return lignes


def moyennes_saisonnieres(serie, mois):
    import numpy as np

    s = np.asarray(serie, dtype=float)
    m = np.asarray(mois, dtype=int)
    return (
        float(s.mean()),
        float(s[np.isin(m, MOIS_ETE)].mean()),
        float(s[np.isin(m, MOIS_HIVER)].mean()),
    )


def plus_longue_plage(valeurs, signe):
    import numpy as np

    v = np.asarray(valeurs, dtype=float)
    bon = (v > 0) if signe > 0 else (v < 0)
    meilleure, debut_m, courante, debut_c = 0, -1, 0, -1
    for k, b in enumerate(bon):
        if b:
            if courante == 0:
                debut_c = k
            courante += 1
            if courante > meilleure:
                meilleure, debut_m = courante, debut_c
        else:
            courante = 0
    return meilleure, debut_m


def signe_constant(valeurs):
    import numpy as np

    v = np.asarray(valeurs, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False, 0
    if (v > 0).all():
        return True, +1
    if (v < 0).all():
        return True, -1
    return False, 0


def tension_aux_points(lo_c, la_c):
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(TENSION).sortby("latitude")
    noms = sorted(
        v
        for v in ds.data_vars
        if "N m**-2" in str(ds[v].attrs.get("units", ""))
        or "N m-2" in str(ds[v].attrs.get("units", ""))
    )
    if len(noms) < 2:
        ds.close()
        raise ValueError("both stress components are missing")
    tx = ds[noms[0]].values
    ty = ds[noms[1]].values
    lats = ds.latitude.values
    lons = ds.longitude.values
    temps = ds["valid_time"].values
    ds.close()

    annees = np.array([int(str(t)[:4]) for t in temps])
    mois = np.array([int(str(t)[5:7]) for t in temps])
    sx = np.empty((len(temps), lo_c.size), dtype=float)
    sy = np.empty((len(temps), lo_c.size), dtype=float)
    for k in range(lo_c.size):
        sx[:, k] = interpoler_serie(tx, lons, lats, lo_c[k], la_c[k])
        sy[:, k] = interpoler_serie(ty, lons, lats, lo_c[k], la_c[k])
    return annees, mois, sx, sy


def indice_aux_points(sx, sy, la_c, normales):
    import numpy as np

    nor = np.asarray(normales, dtype=float)
    la = np.asarray(la_c, dtype=float)
    x = np.asarray(sx, dtype=float)
    y = np.asarray(sy, dtype=float)
    ind = np.empty_like(x)
    for k in range(ind.shape[1]):
        ind[:, k] = indice_vers_le_large(x[:, k], y[:, k], la[k], nor[k])
    return ind


def produire():
    import numpy as np

    if not os.path.exists(TENSION):
        return 1
    os.makedirs(SORTIES, exist_ok=True)

    mer, lons_m, lats_m = masque_cotier()
    afrique = composante_du_bord_sud(~mer, lats_m)
    t1 = dans_la_composante(afrique, lons_m, lats_m, *TEMOIN_AFRIQUE)
    t2 = dans_la_composante(afrique, lons_m, lats_m, *TEMOIN_HORS_AFRIQUE)
    if not (t1 and not t2):
        return 1

    geos = {r: geometrie_cotiere(mer, lons_m, lats_m, r) for r in RAYONS_KM}
    ref = geos[RAYON_DE_REFERENCE]
    lo_c = np.asarray(ref["lon"], dtype=float)
    la_c = np.asarray(ref["lat"], dtype=float)
    for r, g in geos.items():
        if not (
            np.array_equal(np.asarray(g["lon"]), lo_c)
            and np.array_equal(np.asarray(g["lat"]), la_c)
        ):
            return 1
    afr = points_africains(mer, lons_m, lats_m, lo_c, la_c)
    n_afr = int(afr.sum())

    annees, mois, sx, sy = tension_aux_points(lo_c, la_c)

    lignes_rayon, bandes_lignes, diagnostic = [], [], None
    serie_reference = None

    for rayon in RAYONS_KM:
        ind = indice_aux_points(sx, sy, la_c, geos[rayon]["normale"])

        s_afr = ind[:, afr].mean(axis=1)
        s_hors = ind[:, ~afr].mean(axis=1)
        s_tout = ind.mean(axis=1)

        a_afr, e_afr, h_afr = moyennes_saisonnieres(s_afr, mois)
        a_hor, e_hor, h_hor = moyennes_saisonnieres(s_hors, mois)
        a_tou, e_tou, h_tou = moyennes_saisonnieres(s_tout, mois)

        lignes_rayon.append(
            {
                "rayon_km": f"{rayon:.0f}",
                "n_points_total": int(afr.size),
                "n_points_africains": n_afr,
                "annuel_afrique": f"{a_afr:.6f}",
                "ete_afrique": f"{e_afr:.6f}",
                "hiver_afrique": f"{h_afr:.6f}",
                "annuel_hors_afrique": f"{a_hor:.6f}",
                "ete_hors_afrique": f"{e_hor:.6f}",
                "hiver_hors_afrique": f"{h_hor:.6f}",
                "annuel_toutes_cotes": f"{a_tou:.6f}",
                "ete_toutes_cotes": f"{e_tou:.6f}",
                "hiver_toutes_cotes": f"{h_tou:.6f}",
            }
        )

        if abs(rayon - RAYON_DE_REFERENCE) < 1e-9:
            serie_reference = (annees, mois, s_afr)
            bandes = bandes_de_longitude(lo_c[afr])
            diagnostic = diagnostic_des_bandes(lo_c[afr], la_c[afr], bandes)
            for a, b in bandes:
                dans = afr & (lo_c >= a) & (lo_c < b)
                if not dans.any():
                    continue
                s = ind[:, dans].mean(axis=1)
                an, et, hi = moyennes_saisonnieres(s, mois)
                bandes_lignes.append(
                    {
                        "lon_min": f"{a:+.1f}",
                        "lon_max": f"{b:+.1f}",
                        "n_points": int(dans.sum()),
                        "annuel": f"{an:.6f}",
                        "ete": f"{et:.6f}",
                        "hiver": f"{hi:.6f}",
                    }
                )

    ecrire(os.path.join(SORTIES, "ekman_africa_index_by_radius.csv"), lignes_rayon)
    ecrire(os.path.join(SORTIES, "ekman_africa_index_by_band.csv"), bandes_lignes)
    ecrire(
        os.path.join(SORTIES, "ekman_africa_band_diagnostics.csv"),
        [
            {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in d.items()}
            for d in diagnostic
        ],
    )

    verdicts = []
    for saison, cle in (
        ("annuel", "annuel_afrique"),
        ("ete", "ete_afrique"),
        ("hiver", "hiver_afrique"),
    ):
        vals = [float(l[cle]) for l in lignes_rayon]
        const, signe = signe_constant(vals)
        verdicts.append(
            {
                "grandeur": f"{saison}, cote africaine, six rayons",
                "minimum": f"{min(vals):.6f}",
                "maximum": f"{max(vals):.6f}",
                "signe_constant": "oui" if const else "non",
                "signe": {1: "positif", -1: "negatif", 0: "variable"}[signe],
            }
        )
    for saison in ("annuel", "ete", "hiver"):
        vals = [float(l[saison]) for l in bandes_lignes]
        const, signe = signe_constant(vals)
        verdicts.append(
            {
                "grandeur": f"{saison}, par bande de longitude, rayon de reference",
                "minimum": f"{min(vals):.6f}",
                "maximum": f"{max(vals):.6f}",
                "signe_constant": "oui" if const else "non",
                "signe": {1: "positif", -1: "negatif", 0: "variable"}[signe],
            }
        )
    plages = []
    for saison in ("annuel", "ete", "hiver"):
        vals = [float(l[saison]) for l in bandes_lignes]
        for signe, mot in ((+1, "positif"), (-1, "negatif")):
            n_s, k0 = plus_longue_plage(vals, signe)
            plages.append(
                {
                    "saison": saison,
                    "signe": mot,
                    "bandes_de_ce_signe": sum(
                        1 for v in vals if (v > 0) == (signe > 0)
                    ),
                    "plus_longue_plage": n_s,
                    "lon_debut": bandes_lignes[k0]["lon_min"] if k0 >= 0 else "",
                    "lon_fin": (
                        bandes_lignes[k0 + n_s - 1]["lon_max"] if k0 >= 0 else ""
                    ),
                }
            )
    ecrire(os.path.join(SORTIES, "ekman_africa_contiguous_bands.csv"), plages)
    ecrire(os.path.join(SORTIES, "ekman_africa_verdicts.csv"), verdicts)

    for p in plages:
        if not p["plus_longue_plage"]:
            continue

    n = [d["n_points"] for d in diagnostic if d["n_points"]]
    et = [d["etendue_lat"] for d in diagnostic if d["n_points"]]

    annees, mois, s_afr = serie_reference
    with open(
        os.path.join(SORTIES, "ekman_africa_context.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            {
                "definition_de_la_cote": "composante connexe de terre touchant le "
                "bord sud du domaine — aucune latitude",
                "temoins": {
                    "dans_l_afrique": list(TEMOIN_AFRIQUE),
                    "hors_afrique": list(TEMOIN_HORS_AFRIQUE),
                },
                "mailles_de_terre_africaine": int(afrique.sum()),
                "points_africains_par_rayon": {
                    l["rayon_km"]: l["n_points_africains"] for l in lignes_rayon
                },
                "emprise_de_la_cote_africaine": {
                    "lon_min": min(float(b["lon_min"]) for b in bandes_lignes),
                    "lon_max": max(float(b["lon_max"]) for b in bandes_lignes),
                },
                "ce_que_la_cote_contient": "le littoral marocain a l'ouest, "
                "l'Algerie au centre, la Tunisie du "
                "nord a l'est — une frontiere politique "
                "n'est pas une frontiere "
                "oceanographique",
                "bandes": {
                    "largeur_deg": LARGEUR_BANDE,
                    "peuplees": len(n),
                    "effectif_min": min(n),
                    "effectif_max": max(n),
                    "etendue_lat_min": round(min(et), 4),
                    "etendue_lat_max": round(max(et), 4),
                },
                "mois": int(len(mois)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return 0


def ecrire(chemin, lignes):
    if not lignes:
        return
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)


def main(argv):
    if "--produire" in argv:
        return produire()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
