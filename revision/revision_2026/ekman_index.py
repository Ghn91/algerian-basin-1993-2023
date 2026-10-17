"""Offshore Ekman transport along the coast from the ERA5 wind stress, by
coastal segment and for several coastal distances.

Usage: python ekman_index.py --produire
"""

import argparse
import csv
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")

NETCDF_MEDSEA = os.path.join(
    os.environ.get("ABASIN_DATA", "donnees_article"),
    "cmems_mod_med_phy-cur_my_4.2km_P1D-m_uo-vo"
    "_5.00W-10.00E_35.02N-40.98N_1.02m_1993-01-01-2023-12-31.nc",
)
MASQUE_CACHE = os.path.join(DONNEES, "masque_cotier_medsea.npz")
NETCDF_TENSION = os.path.join(DONNEES, "era5_tension_mensuelle_1993_2023.nc")
NETCDF_VENT = os.path.join(DONNEES, "era5_vent_msl_mensuel_1993_2023.nc")

RHO_EAU = 1025.0
OMEGA = 7.2921e-5
RHO_AIR = 1.225
CD = 1.3e-3

M_PAR_DEGRE_LAT = 110574.0
M_PAR_DEGRE_LON = 111320.0

RAYONS_KM = (25.0, 40.0, 60.0, 90.0, 120.0, 150.0)
RAYON_DE_REFERENCE = 60.0

LON_COTE = (-1.0, 9.0)
PAS_SEGMENT_DEG = 1.0


def premiere_couronne(mask_mer):
    m = np.asarray(mask_mer, dtype=bool)
    terre = ~m
    voisin = np.zeros_like(terre)
    voisin[1:, :] |= terre[:-1, :]
    voisin[:-1, :] |= terre[1:, :]
    voisin[:, 1:] |= terre[:, :-1]
    voisin[:, :-1] |= terre[:, 1:]
    return m & voisin


def en_kilometres(lons, lats):
    lon = np.asarray(lons, dtype=float)
    lat = np.asarray(lats, dtype=float)
    x = M_PAR_DEGRE_LON * np.cos(np.radians(lat)) * lon / 1000.0
    y = M_PAR_DEGRE_LAT * lat / 1000.0
    return x, y


def axe_principal(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 2:
        return np.array([np.nan, np.nan]), float("nan")
    p = np.column_stack([x - x.mean(), y - y.mean()])
    c = p.T @ p / max(1, p.shape[0] - 1)
    val, vec = np.linalg.eigh(c)
    tangente = vec[:, -1]
    n = np.linalg.norm(tangente)
    if n == 0 or not np.isfinite(n):
        return np.array([np.nan, np.nan]), float("nan")
    l1, l2 = float(val[-1]), float(val[0])
    nettete = 1.0 - (l2 / l1) if l1 > 0 else float("nan")
    return tangente / n, nettete


def normale_vers_la_mer(tangente, x0, y0, xs_mer, ys_mer):
    t = np.asarray(tangente, dtype=float)
    if not np.all(np.isfinite(t)):
        return np.array([np.nan, np.nan])
    n = np.array([-t[1], t[0]])
    xs = np.asarray(xs_mer, dtype=float)
    ys = np.asarray(ys_mer, dtype=float)
    if xs.size == 0:
        return np.array([np.nan, np.nan])
    vers_mer = np.array([xs.mean() - x0, ys.mean() - y0])
    if np.dot(n, vers_mer) < 0:
        n = -n
    norme = np.linalg.norm(n)
    return n / norme if norme > 0 else np.array([np.nan, np.nan])


def parametre_de_coriolis(lat_deg):
    return 2.0 * OMEGA * np.sin(np.radians(np.asarray(lat_deg, dtype=float)))


def transport_ekman(tau_x, tau_y, lat_deg, rho=RHO_EAU):
    f = parametre_de_coriolis(lat_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(tau_y) / (rho * f), -np.asarray(tau_x) / (rho * f))


def indice_vers_le_large(tau_x, tau_y, lat_deg, normale, rho=RHO_EAU):
    mx, my = transport_ekman(tau_x, tau_y, lat_deg, rho)
    n = np.asarray(normale, dtype=float)
    return mx * n[..., 0] + my * n[..., 1]


def tension_depuis_le_vent(u, v, rho_air=RHO_AIR, cd=CD):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vitesse = np.sqrt(u * u + v * v)
    return rho_air * cd * vitesse * u, rho_air * cd * vitesse * v


def interpoler(champ, lons_src, lats_src, lon_cible, lat_cible):
    lo = np.asarray(lons_src, dtype=float)
    la = np.asarray(lats_src, dtype=float)
    c = np.asarray(champ, dtype=float)
    lon_cible = np.atleast_1d(np.asarray(lon_cible, dtype=float))
    lat_cible = np.atleast_1d(np.asarray(lat_cible, dtype=float))

    i = np.clip(np.searchsorted(lo, lon_cible) - 1, 0, lo.size - 2)
    j = np.clip(np.searchsorted(la, lat_cible) - 1, 0, la.size - 2)
    tx = (lon_cible - lo[i]) / (lo[i + 1] - lo[i])
    ty = (lat_cible - la[j]) / (la[j + 1] - la[j])
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)
    return (
        (1 - tx) * (1 - ty) * c[j, i]
        + tx * (1 - ty) * c[j, i + 1]
        + (1 - tx) * ty * c[j + 1, i]
        + tx * ty * c[j + 1, i + 1]
    )


def geometrie_cotiere(mask_mer, lons, lats, rayon_km):
    cour = premiere_couronne(mask_mer)
    gl, gt = np.meshgrid(np.asarray(lons, float), np.asarray(lats, float))
    x, y = en_kilometres(gl, gt)

    xc, yc = x[cour], y[cour]
    lonc, latc = gl[cour], gt[cour]
    xm, ym = x[mask_mer], y[mask_mer]

    n = xc.size
    tang = np.full((n, 2), np.nan)
    norm = np.full((n, 2), np.nan)
    nett = np.full(n, np.nan)
    voisins = np.zeros(n, dtype=int)

    for k in range(n):
        d = np.hypot(xc - xc[k], yc - yc[k])
        pres = d <= rayon_km
        voisins[k] = int(pres.sum())
        t, q = axe_principal(xc[pres], yc[pres])
        tang[k] = t
        nett[k] = q
        dm = np.hypot(xm - xc[k], ym - yc[k])
        proche = dm <= rayon_km
        norm[k] = normale_vers_la_mer(t, xc[k], yc[k], xm[proche], ym[proche])
    return {
        "lon": lonc,
        "lat": latc,
        "tangente": tang,
        "normale": norm,
        "nettete": nett,
        "voisins": voisins,
        "x": xc,
        "y": yc,
    }


def masque_cotier(chemin_nc=None, cache=MASQUE_CACHE):
    if os.path.exists(cache):
        z = np.load(cache)
        return z["mask"], z["lons"], z["lats"]
    chemin_nc = chemin_nc or NETCDF_MEDSEA
    if not os.path.exists(chemin_nc):
        raise SystemExit(
            f"STOP: neither the cache nor the NetCDF is available.\n  cache  : {cache}\n  source : {chemin_nc}"
        )
    import xarray as xr

    ds = xr.open_dataset(chemin_nc)
    try:
        u0 = ds["uo"].isel(time=0).squeeze().values
        lons = ds["longitude"].values
        lats = ds["latitude"].values
    finally:
        ds.close()
    mask = ~np.isnan(u0)
    os.makedirs(DONNEES, exist_ok=True)
    np.savez_compressed(cache, mask=mask, lons=lons, lats=lats)
    return mask, lons, lats


def _ecrire(nom, entetes, lignes):
    os.makedirs(SORTIES, exist_ok=True)
    c = os.path.join(SORTIES, nom)
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(entetes)
        w.writerows(lignes)


def produire():
    import xarray as xr

    for f in (NETCDF_TENSION, NETCDF_VENT):
        if not os.path.exists(f):
            return 2

    mask, lons_m, lats_m = masque_cotier()

    dt = xr.open_dataset(NETCDF_TENSION)
    dv = xr.open_dataset(NETCDF_VENT)
    try:
        tens = [
            v for v in dt.data_vars if "N m**-2" in str(dt[v].attrs.get("units", ""))
        ]
        est = next(v for v in tens if "ews" in v or "east" in v)
        nord = next(v for v in tens if "nss" in v or "north" in v)
        lo_e, la_e = dt.longitude.values, dt.latitude.values
        ordre = np.argsort(la_e)
        la_e = la_e[ordre]
        taux = dt[est].values[:, ordre, :]
        tauy = dt[nord].values[:, ordre, :]
        u10 = dv["u10"].values[:, ordre, :]
        v10 = dv["v10"].values[:, ordre, :]
        temps = dt[[n for n in ("valid_time", "time") if n in dt.coords][0]]
        mois = temps.dt.month.values
    finally:
        dt.close()
        dv.close()

    lignes_bal, lignes_seg, lignes_pts = [], [], []
    for rayon in RAYONS_KM:
        g = geometrie_cotiere(mask, lons_m, lats_m, rayon)
        dans = (g["lon"] >= LON_COTE[0]) & (g["lon"] <= LON_COTE[1])
        idx = np.where(dans)[0]
        if idx.size == 0:
            continue

        ind = np.full((taux.shape[0], idx.size), np.nan)
        ind_vent = np.full_like(ind, np.nan)
        for p, k in enumerate(idx):
            tx = interpoler_serie(taux, lo_e, la_e, g["lon"][k], g["lat"][k])
            ty = interpoler_serie(tauy, lo_e, la_e, g["lon"][k], g["lat"][k])
            ind[:, p] = indice_vers_le_large(tx, ty, g["lat"][k], g["normale"][k])
            uu = interpoler_serie(u10, lo_e, la_e, g["lon"][k], g["lat"][k])
            vv = interpoler_serie(v10, lo_e, la_e, g["lon"][k], g["lat"][k])
            wx, wy = tension_depuis_le_vent(uu, vv)
            ind_vent[:, p] = indice_vers_le_large(wx, wy, g["lat"][k], g["normale"][k])

        clim = np.array([np.nanmean(ind[mois == m], axis=0) for m in range(1, 13)])
        lignes_bal.append(
            [
                rayon,
                idx.size,
                float(np.nanmean(g["nettete"][idx])),
                float(np.nanmedian(g["voisins"][idx])),
                float(np.nanmean(ind)),
                float(np.nanmean(clim.mean(axis=1))),
                float(np.nanmean(clim.mean(axis=1)[[5, 6, 7]])),
                float(np.nanmean(clim.mean(axis=1)[[11, 0, 1]])),
                float(np.nanmean(ind_vent)),
                (
                    float(np.nanmean(ind_vent) / np.nanmean(ind))
                    if np.nanmean(ind) != 0
                    else float("nan")
                ),
            ]
        )

        if abs(rayon - RAYON_DE_REFERENCE) < 1e-9:
            bornes = np.arange(LON_COTE[0], LON_COTE[1] + 1e-9, PAS_SEGMENT_DEG)
            for b0, b1 in zip(bornes[:-1], bornes[1:]):
                sel = (g["lon"][idx] >= b0) & (g["lon"][idx] < b1)
                if not sel.any():
                    continue
                for m in range(1, 13):
                    lignes_seg.append(
                        [
                            rayon,
                            b0,
                            b1,
                            int(sel.sum()),
                            m,
                            float(np.nanmean(clim[m - 1][sel])),
                            float(np.nanstd(clim[m - 1][sel])),
                        ]
                    )
            for p, k in enumerate(idx):
                lignes_pts.append(
                    [
                        rayon,
                        float(g["lon"][k]),
                        float(g["lat"][k]),
                        float(
                            np.degrees(
                                np.arctan2(g["tangente"][k][1], g["tangente"][k][0])
                            )
                        ),
                        float(
                            np.degrees(
                                np.arctan2(g["normale"][k][1], g["normale"][k][0])
                            )
                        ),
                        float(g["nettete"][k]),
                        int(g["voisins"][k]),
                        float(np.nanmean(ind[:, p])),
                    ]
                )

    _ecrire(
        "ekman_radius_sweep.csv",
        [
            "rayon_km",
            "n_points",
            "nettete_moyenne",
            "voisins_mediane",
            "indice_moyen",
            "indice_moyen_climatologie",
            "indice_JJA",
            "indice_DJF",
            "indice_depuis_vent_moyen",
            "rapport_vent_moyen_sur_tension",
        ],
        lignes_bal,
    )
    _ecrire(
        "ekman_index_by_segment.csv",
        [
            "rayon_km",
            "lon_min",
            "lon_max",
            "n_points",
            "mois",
            "indice_moyen",
            "indice_ecart_type",
        ],
        lignes_seg,
    )
    _ecrire(
        "ekman_coast_geometry.csv",
        [
            "rayon_km",
            "lon",
            "lat",
            "tangente_deg",
            "normale_deg",
            "nettete",
            "voisins",
            "indice_moyen",
        ],
        lignes_pts,
    )
    return 0


def interpoler_serie(champ3d, lons, lats, lon, lat):
    return np.array(
        [
            interpoler(champ3d[t], lons, lats, lon, lat)[0]
            for t in range(champ3d.shape[0])
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--produire", action="store_true")
    a = p.parse_args()
    if a.produire:
        return produire()
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
