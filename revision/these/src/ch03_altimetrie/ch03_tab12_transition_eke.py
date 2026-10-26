"""Pettitt test on the annual EKE anomaly with its false-alarm rate and power,
seasonal cycle, and the zero crossings of the running mean against a
reference of series without change point.

Usage: python ch03_tab12_transition_eke.py [--netcdf PATH] [--tirages N]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))

import pandas as pd
from provenance import sauver_table

NETCDF_DEFAUT = Path(
    "revision/revision_2026/donnees/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc"
)
ANOMALIES = ("ugosa", "vgosa")

ARCHIVE = RACINE / "docs" / "ch03_transition.json"
CM2_PAR_M2 = 1e4
MOIS_PAR_AN = 12
FENETRE = 12
TIRAGES = 400
GRAINE = 2809
GRAINE_PUISSANCE = 2810
ALPHA = 0.05


def pettitt(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return 0, 0.0, 1.0
    signes = np.zeros(n)
    for i in range(n):
        signes[i] = float(np.sum(np.sign(x[i] - x)))
    u = np.cumsum(signes)[:-1]
    k = float(np.max(np.abs(u)))
    indice = int(np.argmax(np.abs(u)))
    p = 2.0 * np.exp(-6.0 * k * k / (n**3 + n**2))
    return indice, k, float(min(p, 1.0))


def ar1(n, phi, sigma=1.0, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    e = rng.normal(scale=sigma * np.sqrt(max(1.0 - phi * phi, 1e-12)), size=n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + e[i]
    return s


def taux_de_rejet_pettitt(
    n,
    phi,
    saut=0.0,
    sigma=1.0,
    position=None,
    tirages=TIRAGES,
    graine=GRAINE,
    alpha=ALPHA,
):
    rng = np.random.default_rng(graine)
    if position is None:
        position = n // 2
    marche = np.zeros(n)
    marche[position:] = 1.0
    rejets = 0
    for _ in range(tirages):
        serie = ar1(n, phi, sigma, rng=rng) + saut * marche
        if pettitt(serie)[2] < alpha:
            rejets += 1
    return rejets / tirages


def moyenne_glissante(y, fenetre=FENETRE):
    y = np.asarray(y, dtype=float)
    demi = fenetre // 2
    rembourre = np.concatenate([np.full(demi, y[0]), y, np.full(demi, y[-1])])
    noyau = np.ones(fenetre) / fenetre
    lisse = np.convolve(rembourre, noyau, mode="same")
    return lisse[demi : demi + len(y)]


def franchissements(y, abscisses, fenetre=FENETRE):
    y = np.asarray(y, dtype=float)
    abscisses = np.asarray(abscisses, dtype=float)
    lisse = moyenne_glissante(y, fenetre)
    signe = np.sign(lisse)
    for i in range(1, len(signe)):
        if signe[i] == 0:
            signe[i] = signe[i - 1]
    change = np.nonzero(np.diff(signe) != 0)[0]
    if len(change) == 0:
        return {
            "nombre": 0,
            "premier": float("nan"),
            "median": float("nan"),
            "dernier": float("nan"),
            "duree_dernier_regime": 1.0,
            "signe_final": float(signe[-1]),
        }
    dates = []
    for i in change:
        a, b = lisse[i], lisse[i + 1]
        part = 0.5 if a == b else a / (a - b)
        dates.append(abscisses[i] + part * (abscisses[i + 1] - abscisses[i]))
    duree = (abscisses[-1] - dates[-1]) / (abscisses[-1] - abscisses[0])
    return {
        "nombre": len(change),
        "premier": float(dates[0]),
        "median": float(np.median(dates)),
        "dernier": float(dates[-1]),
        "duree_dernier_regime": float(duree),
        "signe_final": float(signe[-1]),
    }


def desaisonnaliser(y, mois):
    y = np.asarray(y, dtype=float)
    mois = np.asarray(mois)
    sortie = np.empty_like(y)
    for m in np.unique(mois):
        ou = mois == m
        sortie[ou] = y[ou] - y[ou].mean()
    return sortie


def cycle_saisonnier(y, mois):
    y = np.asarray(y, dtype=float)
    mois = np.asarray(mois)
    lignes = []
    for m in range(1, 13):
        v = y[mois == m]
        lignes.append(
            {
                "mois": m,
                "effectif": len(v),
                "moyenne": float(v.mean()),
                "ecart_type": float(v.std(ddof=1)),
                "minimum": float(v.min()),
                "maximum": float(v.max()),
            }
        )
    return lignes


def bascule(y, annees, charniere):
    y = np.asarray(y, dtype=float)
    annees = np.asarray(annees, dtype=float)
    avant, apres = y[annees < charniere], y[annees >= charniere]
    sigma = float(y.std(ddof=1))
    amplitude = float(apres.mean() - avant.mean())
    return {
        "charniere": float(charniere),
        "n_avant": len(avant),
        "n_apres": len(apres),
        "moyenne_avant": float(avant.mean()),
        "moyenne_apres": float(apres.mean()),
        "amplitude": amplitude,
        "ecart_type": sigma,
        "amplitude_sur_sigma": amplitude / sigma if sigma else float("nan"),
    }


def serie_mensuelle(chemin):
    import xarray as xr

    ds = xr.open_dataset(chemin)
    try:
        absentes = [v for v in ANOMALIES if v not in ds.data_vars]
        if absentes:
            ds.close()
            raise SystemExit(
                f"the file lacks {', '.join(absentes)}.\npresent: {', '.join(sorted(ds.data_vars))}\nThe published velocity anomalies are required; give a file that carries them with --netcdf."
            )
        eke = 0.5 * (ds[ANOMALIES[0]] ** 2 + ds[ANOMALIES[1]] ** 2) * CM2_PAR_M2
        mensuel = eke.resample(time="MS").mean()
        spatial = mensuel.mean(dim=[d for d in mensuel.dims if d != "time"])
        valeurs = np.asarray(spatial.values, dtype=float)
        temps = mensuel["time"].values
        emprise = {
            "longitude_min": float(ds["longitude"].min()),
            "longitude_max": float(ds["longitude"].max()),
            "latitude_min": float(ds["latitude"].min()),
            "latitude_max": float(ds["latitude"].max()),
            "pas_degre": float(abs(np.diff(ds["longitude"].values)[0])),
        }
    finally:
        ds.close()
    mois = np.array([int(str(t)[5:7]) for t in temps])
    an0 = int(str(temps[0])[:4])
    annees = an0 + np.arange(len(valeurs)) / MOIS_PAR_AN
    return valeurs, mois, annees, emprise, temps


def empreinte(chemin, bloc=1 << 20):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(bloc), b""):
            h.update(morceau)
    return h.hexdigest()


def produire(chemin_netcdf, tirages=TIRAGES):
    chemin = Path(chemin_netcdf)
    if not chemin.exists():
        raise SystemExit(
            f"source not found: {chemin}\nGive the TWO-SAT NetCDF path with --netcdf."
        )
    valeurs, mois, annees, emprise, temps = serie_mensuelle(chemin)

    anomalie = desaisonnaliser(valeurs, mois)
    residu_moyenne = float(abs(anomalie.mean()))

    an_de_chaque_mois = np.array([int(str(t)[:4]) for t in temps])
    liste_annees = np.unique(an_de_chaque_mois)
    annuelle = np.array([anomalie[an_de_chaque_mois == a].mean() for a in liste_annees])
    indice, k, p_pettitt = pettitt(annuelle)
    annee_rupture = int(liste_annees[indice])
    r1_annuel = float(np.corrcoef(annuelle[:-1], annuelle[1:])[0, 1])
    sigma_annuel = float(annuelle.std(ddof=1))
    saut_observe = float(annuelle[indice + 1 :].mean() - annuelle[: indice + 1].mean())

    plancher = taux_de_rejet_pettitt(
        len(annuelle), r1_annuel, saut=0.0, sigma=sigma_annuel, tirages=tirages
    )
    puissance = taux_de_rejet_pettitt(
        len(annuelle),
        r1_annuel,
        saut=saut_observe,
        sigma=sigma_annuel,
        position=indice + 1,
        tirages=tirages,
        graine=GRAINE_PUISSANCE,
    )

    reel = franchissements(anomalie, annees)
    milieu = float((annees[0] + annees[-1]) / 2.0)
    rangs = np.arange(len(anomalie), dtype=float)
    pente = float(stats.theilslopes(anomalie, rangs).slope)
    residu = anomalie - pente * rangs
    sigma_residu = float(np.std(residu))
    phi_residu = float(np.corrcoef(residu[:-1], residu[1:])[0, 1])
    rng = np.random.default_rng(GRAINE)
    temoin = {"median": [], "dernier": [], "premier": [], "nombre": []}
    for _ in range(tirages):
        faux = pente * rangs + ar1(len(rangs), phi_residu, sigma_residu, rng=rng)
        faux = faux - faux.mean()
        f = franchissements(faux, annees)
        for cle in temoin:
            temoin[cle].append(f[cle])
    loi = {
        cle: (
            float(np.nanmedian(v)),
            float(np.nanquantile(v, 0.02)),
            float(np.nanquantile(v, 0.98)),
        )
        for cle, v in ((c, np.array(t, dtype=float)) for c, t in temoin.items())
    }
    for cle in ("premier", "median", "dernier", "nombre"):
        m, q2, q98 = loi[cle]

    charnieres = [2005.0, 2008.0]
    bascules = [bascule(anomalie, annees, c) for c in charnieres]

    SCRIPT = "src/ch03_altimetrie/ch03_tab12_transition_eke.py"
    DOM = "bassin-algerien-1993-2023"

    def texte(lignes):
        return pd.DataFrame(
            [
                {c: (f"{v:.4f}" if isinstance(v, float) else v) for c, v in L.items()}
                for L in lignes
            ]
        )

    sauver_table(
        texte(
            [
                {
                    "serie": "anomalie annuelle",
                    "n_annees": len(annuelle),
                    "annee_de_rupture": annee_rupture,
                    "statistique_U": k,
                    "p": p_pettitt,
                    "plancher_sous_h0": plancher,
                    "puissance": puissance,
                    "r1_annuel": r1_annuel,
                    "saut_observe_cm2_s2": saut_observe,
                    "tirages": tirages,
                }
            ]
        ),
        "ch03_tab12",
        "test de pettitt sur l anomalie annuelle et son plancher",
        SCRIPT,
        chemin.name,
        "3.5",
        domaine=DOM,
    )

    sauver_table(
        texte(
            [
                {
                    "statistique": cle,
                    "serie_reelle": reel[cle],
                    "temoin_sans_rupture_median": loi[cle][0],
                    "temoin_quantile_02": loi[cle][1],
                    "temoin_quantile_98": loi[cle][2],
                    "milieu_de_la_periode": milieu,
                    "temoin_r1": phi_residu,
                    "temoin_ecart_type": sigma_residu,
                    "tirages": tirages,
                }
                for cle in ("premier", "median", "dernier", "nombre")
            ]
        ),
        "ch03_tab13",
        "franchissement de zero de la moyenne glissante et son temoin",
        SCRIPT,
        chemin.name,
        "3.5",
        domaine=DOM,
    )

    sauver_table(
        texte(cycle_saisonnier(valeurs, mois)),
        "ch03_tab14",
        "cycle saisonnier de l energie turbulente",
        SCRIPT,
        chemin.name,
        "3.5",
        domaine=DOM,
    )

    sauver_table(
        texte(bascules),
        "ch03_tab15",
        "bascule de l anomalie recalculee par la chaine de these",
        SCRIPT,
        chemin.name,
        "3.5",
        domaine=DOM,
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "fichier": chemin.name,
                "sha256": empreinte(chemin),
                "octets": chemin.stat().st_size,
                "emprise_lue": emprise,
                "premier_mois": str(temps[0])[:7],
                "dernier_mois": str(temps[-1])[:7],
                "premiere_annee": int(liste_annees[0]),
                "derniere_annee": int(liste_annees[-1]),
                "moyenne_de_l_anomalie": residu_moyenne,
                "milieu_de_la_periode": milieu,
                "temoin_r1": phi_residu,
                "temoin_ecart_type": sigma_residu,
                "temoin_pente_par_mois": pente,
                "fenetre_glissante_mois": FENETRE,
                "tirages": tirages,
                "graine_plancher": GRAINE,
                "graine_puissance": GRAINE_PUISSANCE,
                "note": "le p de Pettitt est l'approximation de l'auteur, bornee a 1 ; "
                "le temoin monotone ne porte aucune rupture, par construction",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return reel, loi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    ap.add_argument("--tirages", type=int, default=TIRAGES)
    a = ap.parse_args()
    produire(a.netcdf, tirages=a.tirages)
