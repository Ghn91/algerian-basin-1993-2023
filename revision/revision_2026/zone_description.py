"""Description of the six classes of the zonation: kinematics of the mean field
(velocity, relative vorticity, divergence), and quantities not used by the
classification (TWO-SAT EKE, L4 SST, chlorophyll, depth, eddy detection
density), with the number and position of the connected pieces of each class.
Also writes the numerical sensitivity of the zonation to the precision of the
time mean.

Usage: python zone_description.py --deriver (needs the NetCDF files) | --decrire
"""

import csv
import io
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")
sys.path.insert(0, DOSSIER)

from zonation import (
    _derivee_sans_terre,
    divergence_masquee,
    divergence_publiee,
    metrique_du_pas,
    zonation,
)

DONNEES_ARTICLE = os.environ.get("ABASIN_DATA", "donnees_article")
COURANTS = os.path.join(
    DONNEES_ARTICLE,
    "cmems_mod_med_phy-cur_my_4.2km_P1D-m_uo-vo"
    "_5.00W-10.00E_35.02N-40.98N_1.02m_1993-01-01-2023-12-31.nc",
)
SST = os.path.join(
    DONNEES_ARTICLE,
    "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_multi-vars"
    "_4.96W-9.95E_35.04N-40.96N_1993-01-01-2023-12-31.nc",
)
CHL = os.path.join(
    DONNEES_ARTICLE,
    "cmems_mod_med_bgc-plankton_my_4.2km_P1D-m_chl-phyc"
    "_5.00W-10.00E_35.02N-40.98N_1.02m_1999-01-01-2023-12-31.nc",
)
TWOSAT = os.path.join(
    DONNEES,
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_"
    "4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc",
)
GEBCO = next(
    (
        p
        for p in (
            os.path.join(
                os.environ.get("THESE_DEPOT", "\0"), "data", "raw", "gebco_2026.nc"
            ),
            os.path.join(RACINE, "these", "data", "raw", "gebco_2026.nc"),
        )
        if os.path.isfile(p)
    ),
    "",
)
sys.path.insert(0, DOSSIER)
from eddy_atlas import ATLAS_FICHIERS as ATLAS

MASQUE = os.path.join(DONNEES, "masque_cotier_medsea.npz")
DERIVE = os.path.join(DONNEES, "zones_champs_moyens.npz")
LOCALISATION = os.path.join(SORTIES, "zonation_zone_location.csv")
DIVERGENCE = os.path.join(SORTIES, "zonation_divergence_distribution.csv")
SORTIE = os.path.join(SORTIES, "zone_description.csv")

OMEGA = 7.2921e-5
RAYON_TERRE_KM = 6371.0
TRANCHE_JOURS = 366


def vorticite_relative(u, v, dx, dy):
    return _derivee_sans_terre(v, dx, 1) - _derivee_sans_terre(u, dy, 0)


def parametre_de_coriolis(lats):
    return 2.0 * OMEGA * np.sin(np.radians(np.asarray(lats, dtype=float)))


def plus_proche(source, cible):
    s = np.asarray(source, dtype=float)
    c = np.asarray(cible, dtype=float)
    if np.any(np.diff(s) <= 0):
        raise ValueError("source coordinate must be strictly increasing")
    demi = 0.5 * float(np.median(np.diff(s))) if s.size > 1 else 0.0
    i = np.clip(np.searchsorted(s, c), 1, s.size - 1)
    gauche = s[i - 1]
    droite = s[i]
    j = np.where(np.abs(c - gauche) <= np.abs(c - droite), i - 1, i)
    dehors = (c < s[0] - demi) | (c > s[-1] + demi)
    return np.where(dehors, -1, j)


def regriller(champ, lat_s, lon_s, lat_c, lon_c):
    ilat = plus_proche(lat_s, lat_c)
    ilon = plus_proche(lon_s, lon_c)
    out = np.full((len(lat_c), len(lon_c)), np.nan)
    ok_la = ilat >= 0
    ok_lo = ilon >= 0
    out[np.ix_(ok_la, ok_lo)] = np.asarray(champ, dtype=float)[
        np.ix_(ilat[ok_la], ilon[ok_lo])
    ]
    return out


def moyenne_par_tranches(lire_tranche, n_temps, taille=TRANCHE_JOURS):
    somme = compte = None
    for d in range(0, n_temps, taille):
        a = np.asarray(lire_tranche(d, min(d + taille, n_temps)), dtype=float)
        ok = np.isfinite(a)
        s = np.where(ok, a, 0.0).sum(axis=0)
        c = ok.sum(axis=0)
        somme = s if somme is None else somme + s
        compte = c if compte is None else compte + c
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(compte > 0, somme / np.maximum(compte, 1), np.nan)


def en_celsius(valeurs, unite):
    u = (unite or "").strip().lower()
    if u in ("k", "kelvin", "kelvins"):
        return np.asarray(valeurs, dtype=float) - 273.15
    if u in ("degc", "celsius", "degree_celsius", "degrees_celsius", "°c", "c"):
        return np.asarray(valeurs, dtype=float)
    raise ValueError(f"unknown temperature unit: {unite!r}")


def aire_des_mailles_km2(lats, lons):
    dla = abs(float(np.median(np.diff(lats))))
    dlo = abs(float(np.median(np.diff(lons))))
    k = np.pi / 180.0 * RAYON_TERRE_KM
    ligne = (dla * k) * (dlo * k) * np.cos(np.radians(np.asarray(lats, float)))
    return np.repeat(ligne[:, None], len(lons), axis=1)


def zones_des_points(etiquettes, lats, lons, plat, plon):
    ia = plus_proche(lats, plat)
    io_ = plus_proche(lons, plon)
    z = np.zeros(len(plat), dtype=int)
    ok = (ia >= 0) & (io_ >= 0)
    z[ok] = np.asarray(etiquettes)[ia[ok], io_[ok]]
    return z


def morceaux(masque, glo, gla):
    from scipy import ndimage

    lab, n = ndimage.label(np.asarray(masque, bool), structure=np.ones((3, 3)))
    if n == 0:
        return []
    tailles = np.bincount(lab.ravel())[1:]
    tot = tailles.sum()
    out = []
    for i in np.argsort(tailles)[::-1]:
        sel = lab == i + 1
        out.append(
            (
                float(tailles[i] / tot),
                float(np.median(glo[sel])),
                float(np.median(gla[sel])),
            )
        )
    return out


def resume(valeurs, masque_zone):
    v = np.asarray(valeurs, dtype=float)[masque_zone]
    ok = np.isfinite(v)
    n = int(masque_zone.sum())
    if not ok.any():
        return {"moyenne": float("nan"), "mediane": float("nan"), "couverture": 0.0}
    return {
        "moyenne": float(v[ok].mean()),
        "mediane": float(np.median(v[ok])),
        "couverture": float(ok.sum() / n) if n else 0.0,
    }


def effectifs_reproduits(etiquettes, attendus):
    e = np.asarray(etiquettes)
    ecarts = {}
    for z, n in attendus.items():
        m = int((e == z).sum())
        if m != n:
            ecarts[z] = (m, n)
    if set(np.unique(e[e > 0]).tolist()) != set(attendus):
        ecarts["zones"] = (sorted(np.unique(e[e > 0]).tolist()), sorted(attendus))
    return ecarts


def lire_effectifs_publies():
    with io.open(LOCALISATION, encoding="utf-8") as f:
        return {int(r["zone"]): int(r["n"]) for r in csv.DictReader(f)}


def tranche(var, a, b):
    x = var.isel(time=slice(a, b)).values
    return x.reshape(x.shape[0], *x.shape[-2:])


def accord_des_partitions(a, b):
    from sklearn.metrics import adjusted_rand_score

    a, b = np.asarray(a), np.asarray(b)
    com = (a > 0) & (b > 0)
    return float(adjusted_rand_score(a[com], b[com])), int(com.sum())


def sensibilite_numerique(u, v, u64, v64, e32, e64, ch, ch64, dtype=""):
    ari, n_com = accord_des_partitions(e32, e64)
    with np.errstate(invalid="ignore"):
        du = np.nanmax(np.abs(np.asarray(u, float) - u64)) * 100.0
        dv = np.nanmax(np.abs(np.asarray(v, float) - v64)) * 100.0
    lignes = [
        ["dtype_du_fichier", dtype],
        ["ecart_max_u_cm_s", f"{du:.6e}"],
        ["ecart_max_v_cm_s", f"{dv:.6e}"],
        ["mailles_mer_chemin_figure", int((e32 > 0).sum())],
        ["mailles_mer_chemin_float64", int((e64 > 0).sum())],
        ["mailles_communes", n_com],
        ["indice_de_rand_ajuste", f"{ari:.6f}"],
        ["zones_chemin_figure", int(e32.max())],
        ["zones_chemin_float64", int(e64.max())],
    ]
    for k in sorted(set(ch) | set(ch64)):
        lignes.append([f"critere_k{k}_chemin_figure", f"{ch.get(k, float('nan')):.4f}"])
        lignes.append(
            [f"critere_k{k}_chemin_float64", f"{ch64.get(k, float('nan')):.4f}"]
        )
    for z in range(1, 7):
        lignes.append([f"effectif_z{z}_chemin_figure", int((e32 == z).sum())])
        lignes.append([f"effectif_z{z}_chemin_float64", int((e64 == z).sum())])
    os.makedirs(SORTIES, exist_ok=True)
    chemin = os.path.join(SORTIES, "zone_numerical_sensitivity.csv")
    with io.open(chemin, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grandeur", "valeur"])
        w.writerows(lignes)


def deriver():
    import xarray as xr

    for f in (COURANTS, SST, CHL):
        if not os.path.isfile(f):
            return 2
    with xr.open_dataset(COURANTS) as d:
        lats = d["latitude"].values
        lons = d["longitude"].values
        nt = d.sizes["time"]
        u = d["uo"].mean("time").squeeze().values
        v = d["vo"].mean("time").squeeze().values
        u64 = moyenne_par_tranches(lambda a, b: tranche(d["uo"], a, b), nt)
        v64 = moyenne_par_tranches(lambda a, b: tranche(d["vo"], a, b), nt)
        periode_cur = (
            f"{str(d['time'].values[0])[:10]} -> {str(d['time'].values[-1])[:10]}"
        )
    m = np.load(MASQUE)
    if not (
        m["lats"].shape == lats.shape
        and m["lons"].shape == lons.shape
        and np.allclose(m["lats"], lats)
        and np.allclose(m["lons"], lons)
    ):
        return 1
    etiq_plat, mer, ch, n_opt = zonation(u, v)
    etiquettes = np.zeros(u.shape, dtype=np.int8)
    etiquettes[mer] = etiq_plat
    etiq64_plat, mer64, ch64, n64 = zonation(u64, v64)
    etiq64 = np.zeros(u.shape, dtype=np.int8)
    etiq64[mer64] = etiq64_plat
    sensibilite_numerique(
        u, v, u64, v64, etiquettes, etiq64, ch, ch64, dtype=str(u.dtype)
    )
    ecarts = effectifs_reproduits(etiquettes, lire_effectifs_publies())
    if ecarts:
        return 1

    with xr.open_dataset(SST) as d:
        var = d["analysed_sst"]
        unite = var.attrs.get("units", "")
        nt = d.sizes["time"]
        sst = moyenne_par_tranches(lambda a, b: tranche(var, a, b), nt)
        sst = en_celsius(sst, unite)
        lat_s, lon_s = d["latitude"].values, d["longitude"].values
        periode_sst = (
            f"{str(d['time'].values[0])[:10]} -> {str(d['time'].values[-1])[:10]}"
        )
    if lat_s[0] > lat_s[-1]:
        lat_s, sst = lat_s[::-1], sst[::-1]
    sst_g = regriller(sst, lat_s, lon_s, lats, lons)

    with xr.open_dataset(CHL) as d:
        var = d["chl"]
        unite_chl = var.attrs.get("units", "")
        mois = d["time"].dt.year.values * 100 + d["time"].dt.month.values
        cles = np.unique(mois)
        somme = compte = None
        for k in cles:
            idx = np.nonzero(mois == k)[0]
            a = tranche(var, int(idx[0]), int(idx[-1]) + 1)
            mm = moyenne_par_tranches(lambda a_, b_: a[a_:b_], a.shape[0])
            ok = np.isfinite(mm)
            somme = (
                np.where(ok, mm, 0.0)
                if somme is None
                else somme + np.where(ok, mm, 0.0)
            )
            compte = ok.astype(int) if compte is None else compte + ok
        chl = np.where(compte > 0, somme / np.maximum(compte, 1), np.nan)
        lat_c, lon_c = d["latitude"].values, d["longitude"].values
        periode_chl = (
            f"{str(d['time'].values[0])[:10]} -> {str(d['time'].values[-1])[:10]}"
        )
        n_mois_chl = len(cles)
    meme_grille = (
        lat_c.shape == lats.shape
        and lon_c.shape == lons.shape
        and np.allclose(lat_c, lats)
        and np.allclose(lon_c, lons)
    )
    chl_g = chl if meme_grille else regriller(chl, lat_c, lon_c, lats, lons)

    os.makedirs(DONNEES, exist_ok=True)
    np.savez_compressed(
        DERIVE,
        lats=lats,
        lons=lons,
        etiquettes=etiquettes,
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        sst=sst_g.astype(np.float32),
        chl=chl_g.astype(np.float32),
        periode_courants=periode_cur,
        periode_sst=periode_sst,
        periode_chl=periode_chl,
        n_mois_chl=n_mois_chl,
        unite_sst_source=unite,
        unite_chl=unite_chl,
        sources=np.array(
            [os.path.basename(COURANTS), os.path.basename(SST), os.path.basename(CHL)]
        ),
    )
    return 0


CACHE_EKE = os.path.join(DONNEES, "eke_twosat_moyenne_1993_2023.npz")


def eke_moyenne_twosat():
    import xarray as xr

    if not os.path.isfile(TWOSAT):
        if not os.path.isfile(CACHE_EKE):
            raise SystemExit(
                "STOP  neither the TWO-SAT NetCDF nor the mean-EKE derived file is available"
            )
        c = np.load(CACHE_EKE)
        return c["eke"], c["lats"], c["lons"]
    with xr.open_dataset(TWOSAT) as d:
        nt = d.sizes["time"]
        eke = moyenne_par_tranches(
            lambda a, b: (
                0.5
                * (
                    d["ugosa"].isel(time=slice(a, b)).values ** 2
                    + d["vgosa"].isel(time=slice(a, b)).values ** 2
                )
                * 1e4
            ),
            nt,
        )
        la, lo = d["latitude"].values, d["longitude"].values
    if os.path.isfile(CACHE_EKE):
        c = np.load(CACHE_EKE)
        ecart = float(np.nanmax(np.abs(c["eke"] - eke)))
        if not ecart < 1e-9:
            raise SystemExit(
                f"STOP  the mean-EKE derived file differs from the NetCDF ({ecart})"
            )
    else:
        np.savez_compressed(CACHE_EKE, eke=eke, lats=la, lons=lo)
    return eke, la, lo


def profondeur_gebco(lats, lons):
    import xarray as xr

    if not os.path.isfile(GEBCO):
        return None
    with xr.open_dataset(GEBCO) as d:
        la, lo = d["lat"].values, d["lon"].values
        ilat = plus_proche(la, lats)
        ilon = plus_proche(lo, lons)
        out = np.full((len(lats), len(lons)), np.nan)
        ok_la, ok_lo = ilat >= 0, ilon >= 0
        if ok_la.any() and ok_lo.any():
            z = d["elevation"].values[np.ix_(ilat[ok_la], ilon[ok_lo])]
            out[np.ix_(ok_la, ok_lo)] = np.where(z < 0, -z.astype(float), np.nan)
    return out


def lire_atlas(chemin):
    lat, lon = [], []
    with io.open(chemin, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat.append(float(r["lat"]))
            lon.append(float(r["lon"]))
    return np.array(lat), np.array(lon)


def decrire():
    if not os.path.isfile(DERIVE):
        return 2
    z = np.load(DERIVE, allow_pickle=False)
    lats, lons, etiq = z["lats"], z["lons"], z["etiquettes"].astype(int)
    u, v = z["u"].astype(float), z["v"].astype(float)
    ecarts = effectifs_reproduits(etiq, lire_effectifs_publies())
    if ecarts:
        return 1
    dx, dy = metrique_du_pas(lons, lats)
    vit = np.sqrt(u**2 + v**2) * 100.0
    zeta = vorticite_relative(u, v, dx, dy)
    f = parametre_de_coriolis(lats)[:, None]
    ro = zeta / f
    div_p = divergence_publiee(u, v, dx, dy)
    div_m = divergence_masquee(u, v, dx, dy)

    with io.open(DIVERGENCE, encoding="utf-8") as fh:
        ref = {
            (int(r["zone"]), r["implementation"]): float(r["moyenne"])
            for r in csv.DictReader(fh)
            if r["secteur"] == "toute_la_zone"
        }

    eke, la_e, lo_e = eke_moyenne_twosat()
    eke_g = regriller(eke, la_e, lo_e, lats, lons)
    prof = profondeur_gebco(lats, lons)
    aire = aire_des_mailles_km2(lats, lons)
    atlas = {k: lire_atlas(p) for k, p in ATLAS.items()}
    zones_atlas = {
        k: zones_des_points(etiq, lats, lons, a[0], a[1]) for k, a in atlas.items()
    }
    glo, gla = np.meshgrid(lons, lats)

    lignes = []
    defaut = False
    for c in sorted(np.unique(etiq[etiq > 0])):
        m = etiq == c
        ligne = {
            "zone": int(c),
            "n": int(m.sum()),
            "aire_km2": f"{aire[m].sum():.0f}",
            "lon_q05": f"{np.percentile(glo[m], 5):.2f}",
            "lon_mediane": f"{np.median(glo[m]):.2f}",
            "lon_q95": f"{np.percentile(glo[m], 95):.2f}",
            "lat_q05": f"{np.percentile(gla[m], 5):.2f}",
            "lat_mediane": f"{np.median(gla[m]):.2f}",
            "lat_q95": f"{np.percentile(gla[m], 95):.2f}",
            "u_cm_s": f"{np.nanmean(u[m]) * 100:.4f}",
            "v_cm_s": f"{np.nanmean(v[m]) * 100:.4f}",
            "vitesse_cm_s": f"{np.nanmean(vit[m]):.4f}",
        }
        r = resume(ro, m)
        ligne.update(
            {
                "vorticite_sur_f_moyenne": f"{r['moyenne']:.6f}",
                "vorticite_sur_f_mediane": f"{r['mediane']:.6f}",
                "vorticite_part_cyclonique": f"{np.mean(ro[m][np.isfinite(ro[m])] > 0):.6f}",
                "vorticite_couverture": f"{r['couverture']:.6f}",
            }
        )
        for nom, dv in (("publiee", div_p), ("masquee", div_m)):
            moy = float(np.nanmean(dv[m]))
            ligne[f"divergence_{nom}_s-1"] = f"{moy:.6e}"
            attendu = ref.get((int(c), nom))
            if (
                attendu is None
                or abs(moy - attendu) > 1e-6 * max(abs(attendu), 1e-12) + 1e-14
            ):
                defaut = True
        for nom, champ, fmt in (
            ("eke_twosat_cm2_s-2", eke_g, ".4f"),
            ("sst_degC", z["sst"].astype(float), ".4f"),
            ("chl_mg_m-3", z["chl"].astype(float), ".6f"),
            ("profondeur_m", prof, ".1f"),
        ):
            if champ is None:
                ligne.update(
                    {
                        f"{nom}_moyenne": "",
                        f"{nom}_mediane": "",
                        f"{nom}_couverture": "0.000000",
                    }
                )
                continue
            r = resume(champ, m)
            ligne.update(
                {
                    f"{nom}_moyenne": format(r["moyenne"], fmt),
                    f"{nom}_mediane": format(r["mediane"], fmt),
                    f"{nom}_couverture": f"{r['couverture']:.6f}",
                }
            )
        for i, (part, lo_c, la_c) in enumerate(morceaux(m, glo, gla)[:2], start=1):
            ligne[f"morceau{i}_part"] = f"{part:.6f}"
            ligne[f"morceau{i}_lon_mediane"] = f"{lo_c:.2f}"
            ligne[f"morceau{i}_lat_mediane"] = f"{la_c:.2f}"
        ligne["morceaux"] = len(morceaux(m, glo, gla))
        for k, za in zones_atlas.items():
            n_det = int((za == c).sum())
            ligne[f"detections_{k}"] = n_det
            ligne[f"detections_{k}_par_1e4_km2"] = f"{n_det / aire[m].sum() * 1e4:.4f}"
        lignes.append(ligne)
    hors = {k: int((za == 0).sum()) for k, za in zones_atlas.items()}

    os.makedirs(SORTIES, exist_ok=True)
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    with io.open(
        SORTIE.replace(".csv", "_sources.csv"), "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["grandeur", "valeur"])
        for k in (
            "periode_courants",
            "periode_sst",
            "periode_chl",
            "n_mois_chl",
            "unite_sst_source",
            "unite_chl",
        ):
            w.writerow([k, str(z[k])])
        for k, n in hors.items():
            w.writerow([f"detections_{k}_hors_zone", n])
        w.writerow(["eke_source", os.path.basename(TWOSAT)])
        w.writerow(
            [
                "profondeur_source",
                os.path.basename(GEBCO) if prof is not None else "absente",
            ]
        )
    return 1 if defaut else 0


if __name__ == "__main__":
    if "--deriver" in sys.argv:
        sys.exit(deriver())
    if "--decrire" in sys.argv:
        sys.exit(decrire())
    if "--produire" in sys.argv:
        r = deriver()
        sys.exit(r if r else decrire())
    print(__doc__)
