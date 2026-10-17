"""Hierarchical (Ward) zonation of the 1993-2023 mean MEDSEA surface velocity
field as in the original analysis, the Calinski-Harabasz criterion by number of
classes, the location of each class relative to the coast, and the mean
horizontal divergence computed with and without differencing against land.

Usage: python zonation.py --produire [--netcdf PATH]
"""

import argparse
import csv
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties")

NETCDF_DEFAUT = os.path.join(
    os.environ.get("ABASIN_DATA", "donnees_article"),
    "cmems_mod_med_phy-cur_my_4.2km_P1D-m_uo-vo"
    "_5.00W-10.00E_35.02N-40.98N_1.02m_1993-01-01-2023-12-31.nc",
)

N_ZONES_PUBLIE = 6

K_MAX_BALAYAGE = 20

DIV_Z1_PUBLIE_1E8 = 1.18

SEUILS_BANDE_KM = (20.0, 30.0)

LON_COTE_ALGERIE = (-1.0, 9.0)

M_PAR_DEGRE_LAT = 110574.0
M_PAR_DEGRE_LON = 111320.0


def metrique_du_pas(lons, lats):
    dlat_m = abs(np.diff(lats).mean()) * 111000.0
    dlon_m = abs(np.diff(lons).mean()) * 111000.0 * np.cos(np.radians(np.mean(lats)))
    return dlon_m, dlat_m


def divergence_publiee(u, v, dx, dy):
    u_fg = np.where(np.isnan(u), 0.0, u)
    v_fg = np.where(np.isnan(v), 0.0, v)
    d = np.gradient(u_fg, dx, axis=1) + np.gradient(v_fg, dy, axis=0)
    return np.where(np.isnan(u) | np.isnan(v), np.nan, d)


def _derivee_sans_terre(a, pas, axe):
    a = np.asarray(a, dtype=float)
    apres = np.roll(a, -1, axis=axe)
    avant = np.roll(a, 1, axis=axe)
    fin = [slice(None)] * a.ndim
    fin[axe] = -1
    apres[tuple(fin)] = np.nan
    deb = [slice(None)] * a.ndim
    deb[axe] = 0
    avant[tuple(deb)] = np.nan

    ok_ap = ~np.isnan(apres)
    ok_av = ~np.isnan(avant)
    with np.errstate(invalid="ignore"):
        centree = (apres - avant) / (2.0 * pas)
        vers_av = (apres - a) / pas
        vers_ar = (a - avant) / pas
    d = np.where(
        ok_ap & ok_av,
        centree,
        np.where(ok_ap, vers_av, np.where(ok_av, vers_ar, np.nan)),
    )
    return np.where(np.isnan(a), np.nan, d)


def divergence_masquee(u, v, dx, dy):
    return _derivee_sans_terre(u, dx, 1) + _derivee_sans_terre(v, dy, 0)


def zonation(u, v, k_min=3, k_max=6):
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import calinski_harabasz_score
    from sklearn.preprocessing import StandardScaler

    mask = ~(np.isnan(u) | np.isnan(v))
    X = StandardScaler().fit_transform(np.column_stack([u[mask], v[mask]]))
    Z = linkage(X, method="ward")
    ch = {}
    for n in range(k_min, k_max + 1):
        ch[n] = calinski_harabasz_score(X, fcluster(Z, t=n, criterion="maxclust"))
    n_opt = max(ch, key=ch.get)
    return fcluster(Z, t=n_opt, criterion="maxclust"), mask, ch, n_opt


def lignes_du_critere(scores, borne):
    ks = sorted(scores)
    sommet = max(scores.values()) if scores else None
    lignes = []
    for n in ks:
        voisins = [scores[m] for m in (n - 1, n + 1) if m in scores]
        lignes.append(
            [
                borne,
                n,
                f"{scores[n]:.6f}",
                "oui" if scores[n] == sommet else "non",
                "oui" if n in (ks[0], ks[-1]) else "non",
                "oui" if voisins and scores[n] > max(voisins) else "non",
                f"{scores[n] - max(voisins):.6f}" if voisins else "",
            ]
        )
    return lignes


def distance_a_la_terre_km(mask_mer, lons, lats):
    mask_mer = np.asarray(mask_mer, dtype=bool)
    gl, gt = np.meshgrid(lons, lats)
    x = M_PAR_DEGRE_LON * np.cos(np.radians(gt)) * gl / 1000.0
    y = M_PAR_DEGRE_LAT * gt / 1000.0

    terre = ~mask_mer
    if not terre.any():
        return np.where(mask_mer, np.inf, np.nan)

    xt, yt = x[terre], y[terre]
    xm, ym = x[mask_mer], y[mask_mer]
    d = np.empty(xm.size, dtype=float)
    pas = max(1, int(2_000_000 // max(1, xt.size)))
    for i in range(0, xm.size, pas):
        dx = xm[i : i + pas, None] - xt[None, :]
        dy = ym[i : i + pas, None] - yt[None, :]
        d[i : i + pas] = np.sqrt(dx * dx + dy * dy).min(axis=1)

    out = np.full(mask_mer.shape, np.nan)
    out[mask_mer] = d
    return out


def part_cotiere(distances, seuil_km):
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "part_dans": float("nan"), "part_au_dela": float("nan")}
    dans = int((d <= seuil_km).sum())
    return {
        "n": int(d.size),
        "part_dans": dans / d.size,
        "part_au_dela": 1.0 - dans / d.size,
    }


def composition_de_la_bande(
    etiquettes, distances, longitudes, seuil_km, lon_min, lon_max
):
    e = np.asarray(etiquettes)
    d = np.asarray(distances, dtype=float)
    lo = np.asarray(longitudes, dtype=float)
    dans = np.isfinite(d) & (d <= seuil_km) & (lo >= lon_min) & (lo <= lon_max)
    total = int(dans.sum())
    if total == 0:
        return {"n_bande": 0, "parts": {}}
    parts = {}
    for c in np.unique(e):
        parts[int(c)] = int((dans & (e == c)).sum()) / total
    return {"n_bande": total, "parts": parts}


def resume_divergence(valeurs):
    v = np.asarray(valeurs, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        vide = float("nan")
        return {
            k: vide
            for k in (
                "n",
                "moyenne",
                "ecart_type",
                "q05",
                "q25",
                "mediane",
                "q75",
                "q95",
                "part_positive",
                "moyenne_sur_ecart_type",
            )
        }
    et = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {
        "n": int(v.size),
        "moyenne": float(v.mean()),
        "ecart_type": et,
        "q05": float(np.percentile(v, 5)),
        "q25": float(np.percentile(v, 25)),
        "mediane": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "q95": float(np.percentile(v, 95)),
        "part_positive": float((v > 0).mean()),
        "moyenne_sur_ecart_type": (
            float(v.mean() / et) if et and np.isfinite(et) and et > 0 else float("nan")
        ),
    }


def premiere_couronne(mask_mer):
    m = np.asarray(mask_mer, dtype=bool)
    terre = ~m
    voisin = np.zeros_like(terre)
    voisin[1:, :] |= terre[:-1, :]
    voisin[:-1, :] |= terre[1:, :]
    voisin[:, 1:] |= terre[:, :-1]
    voisin[:, :-1] |= terre[:, 1:]
    return m & voisin


def _ecrire(nom, entetes, lignes):
    os.makedirs(SORTIES, exist_ok=True)
    chemin = os.path.join(SORTIES, nom)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(entetes)
        w.writerows(lignes)


def produire(chemin_nc):
    import xarray as xr

    if not os.path.exists(chemin_nc):
        return 2

    ds = xr.open_dataset(chemin_nc)
    u = ds["uo"].mean("time").squeeze().values
    v = ds["vo"].mean("time").squeeze().values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()

    dx, dy = metrique_du_pas(lons, lats)
    d_pub = divergence_publiee(u, v, dx, dy)
    d_msq = divergence_masquee(u, v, dx, dy)

    etiq, mask, ch, n_opt = zonation(u, v)

    _, _, ch_large, n_large = zonation(u, v, k_min=2, k_max=K_MAX_BALAYAGE)

    if n_opt != N_ZONES_PUBLIE:
        return 1

    dist = distance_a_la_terre_km(mask, lons, lats)
    gl, gt = np.meshgrid(lons, lats)
    couronne = premiere_couronne(mask)

    d_plat = {"publiee": d_pub[mask], "masquee": d_msq[mask]}
    dist_p, lon_p, lat_p = dist[mask], gl[mask], gt[mask]
    cour_p = couronne[mask]

    lignes = []
    for c in range(1, n_opt + 1):
        m = etiq == c
        ligne = [
            c,
            int(m.sum()),
            m.mean(),
            float(np.median(lon_p[m])),
            float(np.median(lat_p[m])),
            float(np.nanmedian(dist_p[m])),
            float(np.nanpercentile(dist_p[m], 5)),
            float(np.nanpercentile(dist_p[m], 95)),
            float(cour_p[m].mean()),
        ]
        for s in SEUILS_BANDE_KM:
            p = part_cotiere(dist_p[m], s)
            ligne += [p["part_dans"], p["part_au_dela"]]
        lignes.append(ligne)
    entetes = [
        "zone",
        "n",
        "part_du_domaine",
        "lon_mediane",
        "lat_mediane",
        "distance_cote_km_mediane",
        "distance_cote_km_q05",
        "distance_cote_km_q95",
        "part_en_premiere_couronne",
    ]
    for s in SEUILS_BANDE_KM:
        entetes += [f"part_dans_{int(s)}km", f"part_au_dela_{int(s)}km"]
    _ecrire("zonation_zone_location.csv", entetes, lignes)

    lignes = []
    for s in SEUILS_BANDE_KM:
        comp = composition_de_la_bande(etiq, dist_p, lon_p, s, *LON_COTE_ALGERIE)
        for c in range(1, n_opt + 1):
            lignes.append(
                [
                    s,
                    LON_COTE_ALGERIE[0],
                    LON_COTE_ALGERIE[1],
                    comp["n_bande"],
                    c,
                    comp["parts"].get(c, 0.0),
                ]
            )
    _ecrire(
        "zonation_coastal_band.csv",
        ["seuil_km", "lon_min", "lon_max", "n_bande", "zone", "part_de_la_bande"],
        lignes,
    )

    lignes = []
    clefs = [
        "n",
        "moyenne",
        "ecart_type",
        "q05",
        "q25",
        "mediane",
        "q75",
        "q95",
        "part_positive",
        "moyenne_sur_ecart_type",
    ]
    for c in range(1, n_opt + 1):
        m = etiq == c
        for nom, dv in d_plat.items():
            for secteur, sel in (
                ("toute_la_zone", m),
                ("premiere_couronne", m & cour_p),
                ("interieur", m & ~cour_p),
            ):
                r = resume_divergence(dv[sel])
                lignes.append([c, nom, secteur] + [r[k] for k in clefs])
    _ecrire(
        "zonation_divergence_distribution.csv",
        ["zone", "implementation", "secteur"] + clefs,
        lignes,
    )

    lignes = []
    for borne, scores in (
        ("depot public (3-6)", ch),
        (f"large (2-{K_MAX_BALAYAGE})", ch_large),
    ):
        lignes += lignes_du_critere(scores, borne)
    _ecrire(
        "zonation_criterion_by_k.csv",
        [
            "balayage",
            "k",
            "critere",
            "maximum_du_balayage",
            "a_la_borne_du_balayage",
            "maximum_local",
            "marge_au_voisin",
        ],
        lignes,
    )

    for c in range(1, n_opt + 1):
        m = etiq == c
    for c in range(1, n_opt + 1):
        m = etiq == c
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--produire",
        action="store_true",
        help="the three tables, from the MEDSEA NetCDF",
    )
    p.add_argument("--netcdf", default=NETCDF_DEFAUT)
    a = p.parse_args()
    if a.produire:
        return produire(a.netcdf)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
