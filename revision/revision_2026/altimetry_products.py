"""ALL-SAT versus TWO-SAT basin-mean EKE: the ALL-SAT product block-averaged
onto the 0.25 deg grid separates the effect of the mapping grid from that of
the constellation; year-by-year ratios and relative trends with block-bootstrap
intervals.

Usage: python altimetry_products.py --produire
"""

import csv
import json
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")

ALLSAT = os.path.join(
    DONNEES,
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_"
    "multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc",
)
TWOSAT = os.path.join(
    DONNEES,
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_"
    "multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_"
    "(1).nc",
)

DERIVE = os.path.join(DONNEES, "eke_mensuelle_trois_champs_1993_2023.csv")

ANNEE_DEBUT, ANNEE_FIN = 1993, 2023
FACTEUR = 2
TOLERANCE_GRILLE = 1e-9

CHAMPS = ("allsat_0125", "allsat_025", "twosat_025")

TIRAGES_BOOTSTRAP = 1000
BLOC_MOIS = 24
GRAINE = 20261005


def eke(u, v):
    import numpy as np

    return 0.5 * (np.asarray(u, dtype=float) ** 2 + np.asarray(v, dtype=float) ** 2)


def moyenner_blocs(champ, facteur=FACTEUR, exiger_complet=True):
    import numpy as np

    a = np.asarray(champ, dtype=float)
    ny, nx = a.shape[-2], a.shape[-1]
    if ny % facteur or nx % facteur:
        raise ValueError(f"moyenner_blocs: grid {ny}x{nx} not divisible by {facteur}")
    forme = a.shape[:-2] + (ny // facteur, facteur, nx // facteur, facteur)
    bloc = a.reshape(forme)
    if exiger_complet:
        return bloc.mean(axis=(-3, -1))
    with np.errstate(invalid="ignore"):
        return np.nanmean(bloc, axis=(-3, -1))


def centres_des_blocs(axe, facteur=FACTEUR):
    import numpy as np

    a = np.asarray(axe, dtype=float)
    if a.size % facteur:
        raise ValueError("centres_des_blocs: size not divisible")
    return a.reshape(-1, facteur).mean(axis=1)


def grilles_coincident(axe_fin, axe_large, facteur=FACTEUR, tolerance=TOLERANCE_GRILLE):
    import numpy as np

    c = centres_des_blocs(axe_fin, facteur)
    l = np.asarray(axe_large, dtype=float)
    if c.size != l.size:
        return False, float("inf")
    ecart = float(np.abs(c - l).max())
    return ecart <= tolerance, ecart


def masque_commun(valide_fin, valide_large, facteur=FACTEUR):
    import numpy as np

    v = np.asarray(valide_fin, dtype=bool)
    ny, nx = v.shape
    bloc = v.reshape(ny // facteur, facteur, nx // facteur, facteur)
    complets = bloc.all(axis=(-3, -1))
    au_moins_un = bloc.any(axis=(-3, -1))
    partiels = int((au_moins_un & ~complets).sum())
    return (
        complets & np.asarray(valide_large, dtype=bool),
        int(complets.sum()),
        partiels,
    )


def pente_de_sen(y, pas_par_decennie=120.0):
    import numpy as np

    a = np.asarray(y, dtype=float)
    n = a.size
    i, j = np.triu_indices(n, k=1)
    pentes = (a[j] - a[i]) / (j - i)
    return float(np.median(pentes)) * pas_par_decennie


def bootstrap_par_blocs(
    y, bloc=BLOC_MOIS, tirages=TIRAGES_BOOTSTRAP, graine=GRAINE, pas_par_decennie=120.0
):
    import numpy as np

    a = np.asarray(y, dtype=float)
    n = a.size
    t = np.arange(n, dtype=float)
    pente = pente_de_sen(a, pas_par_decennie) / pas_par_decennie
    residu = a - pente * t
    rng = np.random.default_rng(graine)
    n_blocs = int(np.ceil(n / bloc))
    pentes = np.empty(tirages)
    for k in range(tirages):
        departs = rng.integers(0, n - bloc + 1, size=n_blocs)
        tire = np.concatenate([residu[d : d + bloc] for d in departs])[:n]
        pentes[k] = pente_de_sen(tire + pente * t, pas_par_decennie)
    return (float(np.quantile(pentes, 0.025)), float(np.quantile(pentes, 0.975)))


def couverture_mesuree(
    n=372, series=200, bloc=BLOC_MOIS, tirages=200, graine=GRAINE, r1=0.0
):
    import numpy as np

    rng = np.random.default_rng(graine)
    dedans = 0
    for k in range(series):
        bruit = rng.normal(size=n)
        if r1:
            y = np.empty(n)
            y[0] = bruit[0]
            for i in range(1, n):
                y[i] = r1 * y[i - 1] + bruit[i]
        else:
            y = bruit
        bas, haut = bootstrap_par_blocs(
            y, bloc=bloc, tirages=tirages, graine=int(rng.integers(1, 10**6))
        )
        dedans += int(bas <= 0 <= haut)
    return dedans / float(series)


def rapport(a, b):
    return float(a) / float(b) if b else float("nan")


def serie_mensuelle(
    chemin,
    masque_fin=None,
    masque_large=None,
    degrader=False,
    var_u="ugosa",
    var_v="vgosa",
):
    import numpy as np
    import xarray as xr

    mois, valeurs = [], []
    ds = xr.open_dataset(chemin)
    try:
        for annee in range(ANNEE_DEBUT, ANNEE_FIN + 1):
            tranche = ds.sel(time=str(annee))
            u = tranche[var_u].values
            v = tranche[var_v].values
            temps = tranche["time"].values
            if degrader:
                u = moyenner_blocs(u)
                v = moyenner_blocs(v)
            champ = eke(u, v)
            m = masque_large if degrader or masque_fin is None else masque_fin
            for k in range(1, 13):
                ou = np.array([int(str(t)[5:7]) == k for t in temps])
                if not ou.any():
                    continue
                moyen = champ[ou].mean(axis=0)
                valeurs.append(float(moyen[m].mean()))
                mois.append((annee, k))
    finally:
        ds.close()
    return mois, np.array(valeurs)


def produire():
    import numpy as np
    import xarray as xr

    for chemin in (ALLSAT, TWOSAT):
        if not os.path.exists(chemin):
            return 1
    os.makedirs(SORTIES, exist_ok=True)

    a = xr.open_dataset(ALLSAT)
    t = xr.open_dataset(TWOSAT)

    for nom, fin, large in (
        ("latitude", a.latitude.values, t.latitude.values),
        ("longitude", a.longitude.values, t.longitude.values),
    ):
        ok, ecart = grilles_coincident(fin, large)
        if not ok:
            a.close()
            t.close()
            return 1

    valide_fin = np.isfinite(a["ugosa"].isel(time=0).values)
    valide_large = np.isfinite(t["ugosa"].isel(time=0).values)
    masque, complets, partiels = masque_commun(valide_fin, valide_large)
    n_fin = int(valide_fin.sum())
    a.close()
    t.close()

    masque_fin = np.repeat(np.repeat(masque, FACTEUR, axis=0), FACTEUR, axis=1)
    masque_fin = masque_fin & valide_fin

    series = {}
    mois, series["allsat_0125"] = serie_mensuelle(
        ALLSAT, masque_fin=masque_fin, masque_large=masque
    )
    _, series["allsat_025"] = serie_mensuelle(
        ALLSAT, masque_large=masque, degrader=True
    )
    _, series["twosat_025"] = serie_mensuelle(
        TWOSAT, masque_fin=masque, masque_large=masque
    )

    for nom in CHAMPS:
        if len(series[nom]) != len(mois):
            return 1

    with open(DERIVE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["annee", "mois"] + list(CHAMPS))
        for k, (an, mo) in enumerate(mois):
            w.writerow([an, mo] + [f"{series[n][k]:.10e}" for n in CHAMPS])

    moy = {n: float(series[n].mean()) for n in CHAMPS}
    lignes = [
        {
            "effet": "resolution de cartographie",
            "de": "allsat_0125",
            "vers": "allsat_025",
            "eke_de": f"{moy['allsat_0125']:.10e}",
            "eke_vers": f"{moy['allsat_025']:.10e}",
            "rapport": f"{rapport(moy['allsat_025'], moy['allsat_0125']):.6f}",
            "lecture": "meme constellation, maille deux fois plus large",
        },
        {
            "effet": "constellation",
            "de": "allsat_025",
            "vers": "twosat_025",
            "eke_de": f"{moy['allsat_025']:.10e}",
            "eke_vers": f"{moy['twosat_025']:.10e}",
            "rapport": f"{rapport(moy['twosat_025'], moy['allsat_025']):.6f}",
            "lecture": "meme maille, deux missions au lieu de toutes",
        },
        {
            "effet": "les deux ensemble (produits publies)",
            "de": "allsat_0125",
            "vers": "twosat_025",
            "eke_de": f"{moy['allsat_0125']:.10e}",
            "eke_vers": f"{moy['twosat_025']:.10e}",
            "rapport": f"{rapport(moy['twosat_025'], moy['allsat_0125']):.6f}",
            "lecture": "ce qu'on mesurerait en comparant les deux produits tels "
            "qu'ils sont publies — la somme des deux effets",
        },
    ]
    ecrire(os.path.join(SORTIES, "products_attenuation.csv"), lignes)

    annees = sorted({an for an, _ in mois})
    par_annee = []
    for an in annees:
        ou = np.array([a0 == an for a0, _ in mois])
        m = {n: float(series[n][ou].mean()) for n in CHAMPS}
        par_annee.append(
            {
                "annee": an,
                "eke_allsat_0125": f"{m['allsat_0125']:.10e}",
                "eke_allsat_025": f"{m['allsat_025']:.10e}",
                "eke_twosat_025": f"{m['twosat_025']:.10e}",
                "rapport_resolution": f"{rapport(m['allsat_025'], m['allsat_0125']):.6f}",
                "rapport_constellation": f"{rapport(m['twosat_025'], m['allsat_025']):.6f}",
                "rapport_total": f"{rapport(m['twosat_025'], m['allsat_0125']):.6f}",
            }
        )
    ecrire(os.path.join(SORTIES, "products_ratios_by_year.csv"), par_annee)
    rc = np.array([float(l["rapport_constellation"]) for l in par_annee])

    r1_moyen = float(
        np.mean([np.corrcoef(series[n][1:], series[n][:-1])[0, 1] for n in CHAMPS])
    )
    couverture = couverture_mesuree(series=120, tirages=200, r1=r1_moyen)

    tend = []
    for nom in CHAMPS:
        p = pente_de_sen(series[nom])
        bas, haut = bootstrap_par_blocs(series[nom])
        tend.append(
            {
                "champ": nom,
                "eke_moyenne": f"{moy[nom]:.10e}",
                "pente_par_decennie": f"{p:.10e}",
                "pente_relative_par_decennie": f"{rapport(p, moy[nom]):.6f}",
                "intervalle_bas": f"{bas:.10e}",
                "intervalle_haut": f"{haut:.10e}",
                "zero_dedans": "oui" if bas <= 0 <= haut else "non",
                "couverture_mesuree_sous_le_nul": f"{couverture:.4f}",
                "r1_de_la_serie": f"{float(np.corrcoef(series[nom][1:], series[nom][:-1])[0, 1]):.6f}",
                "bloc_mois": BLOC_MOIS,
                "tirages": TIRAGES_BOOTSTRAP,
            }
        )
    ecrire(os.path.join(SORTIES, "products_trends.csv"), tend)

    croisees = {}
    for i, n1 in enumerate(CHAMPS):
        for n2 in CHAMPS[i + 1 :]:
            croisees[f"{n1}_{n2}"] = round(
                float(np.corrcoef(series[n1], series[n2])[0, 1]), 6
            )

    with open(
        os.path.join(SORTIES, "products_context.json"),
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        json.dump(
            {
                "mailles_de_mer_allsat_0125": n_fin,
                "blocs_complets": complets,
                "blocs_partiels_ecartes": partiels,
                "mailles_de_mer_twosat_025": int(valide_large.sum()),
                "mailles_du_domaine_commun": int(masque.sum()),
                "mois": len(mois),
                "variable_comparee": "ugosa/vgosa — anomalies, l'ALL-SAT "
                "telecharge ne portant pas ugos/vgos",
                "correlations_des_series": croisees,
                "rapport_constellation_min": round(float(rc.min()), 6),
                "rapport_constellation_max": round(float(rc.max()), 6),
                "bloc_mois": BLOC_MOIS,
                "tirages_bootstrap": TIRAGES_BOOTSTRAP,
                "couverture_mesuree_sous_le_nul": round(couverture, 4),
                "r1_moyen_des_series": round(r1_moyen, 6),
                "graine": GRAINE,
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
