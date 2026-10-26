"""Coastal minus offshore summer SST along the African coast (L4 reprocessed
product), and its relation to the offshore Ekman transport: interannual
correlations by coastal band and a spatial test against circularly shifted
bands.

Usage: python coastal_sst.py --preparer [--source PATH] | --produire
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
    RAYON_DE_REFERENCE,
    en_kilometres,
    geometrie_cotiere,
    indice_vers_le_large,
    interpoler_serie,
    masque_cotier,
)
from ekman_african_coast import (
    bandes_de_longitude,
    diagnostic_des_bandes,
    points_africains,
)

SST_PAR_DEFAUT = os.path.join(
    RACINE, "these", "data", "raw", "medsea_sst-l4_1993-2024.nc"
)
DERIVE = os.path.join(DONNEES, "sst_cote_large_mensuelle_1993_2023.csv")
TENSION = os.path.join(DONNEES, "era5_tension_mensuelle_1993_2023.nc")

ANNEE_DEBUT, ANNEE_FIN = 1993, 2023

BANDES_COTE_KM = ((0.0, 15.0), (0.0, 25.0), (0.0, 40.0))
BANDES_LARGE_KM = ((60.0, 120.0), (75.0, 150.0), (100.0, 200.0))

MOIS_ETE = (6, 7, 8)
MOIS_HIVER = (12, 1, 2)
RETARDS = (0, 1)

SIGNE_ATTENDU = -1

LARGEURS_DE_BANDE_DEG = (1.0, 0.5)

PAIRE_SPATIALE = ((0.0, 25.0), (75.0, 150.0))


def anomalie_mensuelle(serie, mois):
    import numpy as np

    a = np.asarray(serie, dtype=float).copy()
    m = np.asarray(mois, dtype=int)
    for k in range(1, 13):
        ou = m == k
        if ou.any():
            a[ou] = a[ou] - a[ou].mean()
    return a


def correlation(a, b):
    import numpy as np

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    bons = np.isfinite(x) & np.isfinite(y)
    x, y = x[bons] - x[bons].mean(), y[bons] - y[bons].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return (float((x * y).sum() / den) if den > 0 else float("nan")), int(bons.sum())


def p_bilateral(r, n):
    import numpy as np
    from scipy import stats

    if not np.isfinite(r) or n < 4 or abs(r) >= 1:
        return float("nan")
    t = r * np.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), n - 2))


def autocorrelation_1(x):
    import numpy as np

    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    a = a - a.mean()
    den = (a * a).sum()
    return float((a[1:] * a[:-1]).sum() / den) if den > 0 else 0.0


def verdict_du_signe(r, attendu=SIGNE_ATTENDU):
    import numpy as np

    if not np.isfinite(r):
        return "sans objet"
    if r == 0:
        return "nul"
    return "conforme" if (r < 0) == (attendu < 0) else "CONTRAIRE"


def segments_hors_emprise(lons_cote, lats_cote, emprise):
    import numpy as np

    lo = np.asarray(lons_cote, dtype=float)
    la = np.asarray(lats_cote, dtype=float)
    sud, nord, ouest, est = emprise
    dedans = (la >= sud) & (la <= nord) & (lo >= ouest) & (lo <= est)
    return int(dedans.sum()), int((~dedans).sum())


def preparer(source=None):
    import numpy as np
    import xarray as xr

    source = source or SST_PAR_DEFAUT
    if not os.path.isfile(source):
        return 1
    os.makedirs(DONNEES, exist_ok=True)

    mask, lons_m, lats_m = masque_cotier()
    geo = geometrie_cotiere(mask, lons_m, lats_m, RAYON_DE_REFERENCE)
    lons_tous = np.asarray(geo["lon"], dtype=float)
    lats_tous = np.asarray(geo["lat"], dtype=float)

    afr = points_africains(mask, lons_m, lats_m, lons_tous, lats_tous)
    lons_cote = lons_tous[afr]
    lats_cote = lats_tous[afr]

    ds = xr.open_dataset(source)
    nom_lat = "latitude" if "latitude" in ds.coords else "lat"
    nom_lon = "longitude" if "longitude" in ds.coords else "lon"
    lats = ds[nom_lat].values
    lons = ds[nom_lon].values
    emprise = (
        float(lats.min()),
        float(lats.max()),
        float(lons.min()),
        float(lons.max()),
    )
    dedans, dehors = segments_hors_emprise(lons_cote, lats_cote, emprise)
    if dedans == 0:
        ds.close()
        return 1

    LO, LA = np.meshgrid(lons, lats)
    dist = np.empty(LO.shape, dtype=float)
    xc, yc = en_kilometres(lons_cote, lats_cote)
    for i in range(LO.shape[0]):
        x, y = en_kilometres(LO[i], LA[i])
        dist[i] = np.sqrt(
            (x[:, None] - xc[None, :]) ** 2 + (y[:, None] - yc[None, :]) ** 2
        ).min(axis=1)

    mer = np.isfinite(ds["analysed_sst"].isel(time=0).values)

    (c0s, c1s), (l0s, l1s) = PAIRE_SPATIALE
    plans = {}
    for largeur in LARGEURS_DE_BANDE_DEG:
        bornes = bandes_de_longitude(lons_cote, largeur)
        diag = diagnostic_des_bandes(lons_cote, lats_cote, bornes)
        peuplees = [(b, d) for b, d in zip(bornes, diag) if d["n_points"]]
        cles = []
        for (a, b), d in peuplees:
            dans_lon = (LO >= a) & (LO < b)
            mc = mer & dans_lon & (dist >= c0s) & (dist < c1s)
            ml = mer & dans_lon & (dist >= l0s) & (dist < l1s)
            cle = f"b{largeur:g}_{a:+.1f}"
            cles.append(
                (cle, a, b, d["n_points"], mc, ml, int(mc.sum()), int(ml.sum()))
            )
        plans[largeur] = cles
        vides = sum(1 for c in cles if c[6] == 0 or c[7] == 0)

    lignes = []
    for annee in range(ANNEE_DEBUT, ANNEE_FIN + 1):
        tranche = ds["analysed_sst"].sel(time=str(annee))
        sst = tranche.values
        temps = tranche["time"].values
        mois = np.array([int(str(t)[5:7]) for t in temps])
        for k in range(1, 13):
            ou = mois == k
            if not ou.any():
                continue
            tranche_mois = sst[ou]
            vides = ~np.isfinite(tranche_mois).any(axis=0)
            if (vides & mer).any():
                ds.close()
                return 1
            champ = np.full(mer.shape, np.nan)
            champ[mer] = np.nanmean(tranche_mois[:, mer], axis=0)
            ligne = {"annee": annee, "mois": k}
            for c0, c1 in BANDES_COTE_KM:
                m = mer & (dist >= c0) & (dist < c1)
                ligne[f"cote_{c0:.0f}_{c1:.0f}"] = (
                    f"{float(np.nanmean(champ[m])):.6f}" if m.any() else ""
                )
            for l0, l1 in BANDES_LARGE_KM:
                m = mer & (dist >= l0) & (dist < l1)
                ligne[f"large_{l0:.0f}_{l1:.0f}"] = (
                    f"{float(np.nanmean(champ[m])):.6f}" if m.any() else ""
                )
            for largeur in LARGEURS_DE_BANDE_DEG:
                for cle, _, _, _, mc, ml, nc, nl in plans[largeur]:
                    ligne[f"{cle}_cote"] = (
                        f"{float(np.nanmean(champ[mc])):.6f}" if nc else ""
                    )
                    ligne[f"{cle}_large"] = (
                        f"{float(np.nanmean(champ[ml])):.6f}" if nl else ""
                    )
            lignes.append(ligne)
    ds.close()

    with open(DERIVE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)

    with open(
        os.path.join(SORTIES, "sst_context.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(
            {
                "source": source,
                "produit": "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021, "
                "0,05°, journalier, GAP-FREE donc RECONSTRUIT",
                "emprise_sst": {
                    "sud": emprise[0],
                    "nord": emprise[1],
                    "ouest": emprise[2],
                    "est": emprise[3],
                },
                "definition_de_la_cote": "composantes connexes de la terre touchant le bord sud (ekman_african_coast) — aucune latitude",
                "points_de_cote_africains": int(lons_cote.size),
                "points_de_cote_du_domaine": int(lons_tous.size),
                "points_de_cote_dedans": dedans,
                "points_de_cote_dehors": dehors,
                "mailles_sst_valides": int(mer.sum()),
                "bandes_cote_km": [list(b) for b in BANDES_COTE_KM],
                "bandes_large_km": [list(b) for b in BANDES_LARGE_KM],
                "signe_attendu": SIGNE_ATTENDU,
                "test_spatial": {
                    "paire_de_distances": [
                        list(PAIRE_SPATIALE[0]),
                        list(PAIRE_SPATIALE[1]),
                    ],
                    "largeurs_de_bande_deg": list(LARGEURS_DE_BANDE_DEG),
                    "bandes_par_largeur": {
                        f"{l:g}": len(plans[l]) for l in LARGEURS_DE_BANDE_DEG
                    },
                    "plancher_par_largeur": {
                        f"{l:g}": round(1.0 / len(plans[l]), 6)
                        for l in LARGEURS_DE_BANDE_DEG
                    },
                    "prediction": "l'anomalie froide d'ete doit etre plus marquee "
                    "entre 0 et +7 E, ou l'indice d'Ekman d'ete est "
                    "positif sur sept bandes contigues",
                    "temoin": "decalage circulaire des bandes, qui conserve la "
                    "contiguite — les rebattre la detruirait et rendrait "
                    "un plancher optimiste",
                },
                "mois": len(lignes),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return 0


def indice_ekman_mensuel():
    import numpy as np
    import xarray as xr

    mask, lons_m, lats_m = masque_cotier()
    geo = geometrie_cotiere(mask, lons_m, lats_m, RAYON_DE_REFERENCE)

    ds = xr.open_dataset(TENSION).sortby("latitude")
    noms = [
        v
        for v in ds.data_vars
        if "N m**-2" in str(ds[v].attrs.get("units", ""))
        or "N m-2" in str(ds[v].attrs.get("units", ""))
    ]
    if len(noms) < 2:
        ds.close()
        raise ValueError("both stress components are missing")
    est, nord = sorted(noms)
    tx = ds[est].values
    ty = ds[nord].values
    lats = ds.latitude.values
    lons = ds.longitude.values
    temps = ds["valid_time"].values
    ds.close()

    annees = np.array([int(str(t)[:4]) for t in temps])
    mois = np.array([int(str(t)[5:7]) for t in temps])

    lo_t = np.asarray(geo["lon"], dtype=float)
    la_t = np.asarray(geo["lat"], dtype=float)
    nor_t = np.asarray(geo["normale"], dtype=float)
    afr = points_africains(mask, lons_m, lats_m, lo_t, la_t)
    lo_c, la_c, nor = lo_t[afr], la_t[afr], nor_t[afr]

    par_point = np.empty((len(temps), lo_c.size), dtype=float)
    for k in range(lo_c.size):
        sx = interpoler_serie(tx, lons, lats, lo_c[k], la_c[k])
        sy = interpoler_serie(ty, lons, lats, lo_c[k], la_c[k])
        par_point[:, k] = indice_vers_le_large(sx, sy, la_c[k], nor[k])
    return annees, mois, par_point.mean(axis=1), par_point, lo_c, la_c


def indice_par_bande(par_point, lo_c, mois, largeur):
    import numpy as np

    m = np.asarray(mois, dtype=int)
    ete = np.isin(m, MOIS_ETE)
    sortie = []
    for a, b in bandes_de_longitude(lo_c, largeur):
        dans = (lo_c >= a) & (lo_c < b)
        if not dans.any():
            continue
        sortie.append(
            (f"b{largeur:g}_{a:+.1f}", a, float(par_point[ete][:, dans].mean()))
        )
    return sortie


def correlation_decalee(x, y, decalage):
    import numpy as np

    return correlation(
        np.asarray(x, dtype=float), np.roll(np.asarray(y, dtype=float), decalage)
    )


def plancher_circulaire(n):
    return 1.0 / float(n) if n > 0 else float("nan")


def p_par_decalage(x, y):
    import numpy as np

    n = len(x)
    r0 = correlation_decalee(x, y, 0)[0]
    rs = np.array([correlation_decalee(x, y, d)[0] for d in range(n)])
    au_moins = int((rs <= r0 + 1e-15).sum())
    return r0, au_moins / float(n), n


def produire():
    import numpy as np

    if not os.path.exists(DERIVE):
        return 1
    if not os.path.exists(TENSION):
        return 1
    os.makedirs(SORTIES, exist_ok=True)

    with open(DERIVE, newline="", encoding="utf-8") as f:
        sst = list(csv.DictReader(f))
    an_sst = np.array([int(l["annee"]) for l in sst])
    mo_sst = np.array([int(l["mois"]) for l in sst])

    an_ek, mo_ek, ind, par_point, lo_c, la_c = indice_ekman_mensuel()
    garde = (an_ek >= ANNEE_DEBUT) & (an_ek <= ANNEE_FIN)
    an_ek, mo_ek, ind = an_ek[garde], mo_ek[garde], ind[garde]
    par_point = par_point[garde]
    if not (np.array_equal(an_ek, an_sst) and np.array_equal(mo_ek, mo_sst)):
        return 1

    ind_a = anomalie_mensuelle(ind, mo_sst)

    lignes = []
    for c0, c1 in BANDES_COTE_KM:
        for l0, l1 in BANDES_LARGE_KM:
            kc, kl = f"cote_{c0:.0f}_{c1:.0f}", f"large_{l0:.0f}_{l1:.0f}"
            diff = np.array(
                [
                    float(l[kc]) - float(l[kl]) if l[kc] and l[kl] else np.nan
                    for l in sst
                ]
            )
            diff_a = anomalie_mensuelle(diff, mo_sst)
            for nom_saison, mois_saison in (
                ("ete", MOIS_ETE),
                ("hiver", MOIS_HIVER),
                ("toute l'annee", tuple(range(1, 13))),
            ):
                for retard in RETARDS:
                    ou = np.isin(mo_sst, mois_saison)
                    i = (
                        ind_a[: len(ind_a) - retard][ou[: len(ou) - retard]]
                        if retard
                        else ind_a[ou]
                    )
                    d = (
                        diff_a[retard:][ou[: len(ou) - retard]]
                        if retard
                        else diff_a[ou]
                    )
                    r, n = correlation(i, d)
                    lignes.append(
                        {
                            "bande_cote_km": f"{c0:.0f}-{c1:.0f}",
                            "bande_large_km": f"{l0:.0f}-{l1:.0f}",
                            "saison": nom_saison,
                            "retard_mois": retard,
                            "n_mois": n,
                            "difference_moyenne_C": f"{float(np.nanmean(diff[ou])):.6f}",
                            "r": f"{r:.6f}",
                            "p_bilateral": f"{p_bilateral(r, n):.6f}",
                            "signe_attendu": SIGNE_ATTENDU,
                            "verdict_du_signe": verdict_du_signe(r),
                        }
                    )

    ecrire(os.path.join(SORTIES, "sst_correlations.csv"), lignes)

    ete = [l for l in lignes if l["saison"] == "ete" and l["retard_mois"] == 0]
    conformes = sum(1 for l in ete if l["verdict_du_signe"] == "conforme")

    ete_mois = np.isin(mo_sst, MOIS_ETE)
    spatial = []
    for largeur in LARGEURS_DE_BANDE_DEG:
        bandes_ek = indice_par_bande(par_point, lo_c, mo_sst, largeur)
        x, y, cles = [], [], []
        for cle, _, ind_ete in bandes_ek:
            kc, kl = f"{cle}_cote", f"{cle}_large"
            if kc not in sst[0] or kl not in sst[0]:
                continue
            d = np.array(
                [
                    float(l[kc]) - float(l[kl]) if l[kc] and l[kl] else np.nan
                    for l in sst
                ]
            )
            if not np.isfinite(d).any():
                continue
            d_a = anomalie_mensuelle(d, mo_sst)
            ecart = (
                float(np.nanmean(d_a[ete_mois]))
                if np.isfinite(d_a[ete_mois]).any()
                else np.nan
            )
            brut = (
                float(np.nanmean(d[ete_mois]))
                if np.isfinite(d[ete_mois]).any()
                else np.nan
            )
            if not (np.isfinite(brut) and np.isfinite(ind_ete)):
                continue
            x.append(ind_ete)
            y.append(brut)
            cles.append(cle)
            spatial.append(
                {
                    "largeur_bande_deg": f"{largeur:g}",
                    "bande": cle,
                    "indice_ekman_ete": f"{ind_ete:.6f}",
                    "ecart_cote_large_ete_C": f"{brut:.6f}",
                    "anomalie_ete_C": f"{ecart:.6f}" if np.isfinite(ecart) else "",
                }
            )
        if len(x) < 3:
            continue
        r0, p, n = p_par_decalage(x, y)
        sol = plancher_circulaire(n)
        spatial.append(
            {
                "largeur_bande_deg": f"{largeur:g}",
                "bande": "— VERDICT —",
                "indice_ekman_ete": f"n={n}",
                "ecart_cote_large_ete_C": f"r={r0:.6f}",
                "anomalie_ete_C": f"p={p:.6f} plancher={sol:.6f}",
            }
        )

    ecrire(os.path.join(SORTIES, "sst_spatial_test.csv"), spatial)

    with open(
        os.path.join(SORTIES, "sst_verdict.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(
            {
                "couples_de_bandes": len(ete),
                "signe_conforme_en_ete_retard_0": conformes,
                "r_ete_retard_0": [float(l["r"]) for l in ete],
                "p_ete_retard_0": [float(l["p_bilateral"]) for l in ete],
                "r1_anomalie_indice": round(autocorrelation_1(ind_a), 6),
                "signe_attendu": SIGNE_ATTENDU,
                "puissance": "trente-et-un etes, donc un effectif de trente par "
                "mois civil : une correlation de faible amplitude ne "
                "serait pas detectable, et l'absence de signature ne "
                "contredirait pas l'indice",
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
    source = None
    if "--source" in argv:
        source = argv[argv.index("--source") + 1]
    if "--preparer" in argv:
        return preparer(source)
    if "--produire" in argv:
        return produire()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
