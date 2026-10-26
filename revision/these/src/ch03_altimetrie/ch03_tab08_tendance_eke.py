"""Trend of basin-mean EKE (TWO-SAT) by six test procedures (ordinary
Mann-Kendall, pre-whitening, trend-free pre-whitening; raw and deseasonalised
series), each with its rejection rate under the null and its power.

Usage: python ch03_tab08_tendance_eke.py [--netcdf PATH] [--tirages N]
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

ARCHIVE = RACINE / "docs" / "ch03_tendance.json"
CM2_PAR_M2 = 1e4
MOIS_PAR_AN = 12
TIRAGES_H0 = 400
GRAINE_H0 = 1793
ALPHA = 0.05


def mann_kendall(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = 0.0
    for i in range(n - 1):
        s += float(np.sum(np.sign(x[i + 1 :] - x[i])))
    _, effectifs = np.unique(x, return_counts=True)
    correction = float(np.sum(effectifs * (effectifs - 1) * (2 * effectifs + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - correction) / 18.0
    if var_s <= 0:
        return s, 0.0, 0.0, 1.0, var_s
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return s, float(tau), float(z), float(p), float(var_s)


def pente_sen(y, t):
    return float(
        stats.theilslopes(np.asarray(y, dtype=float), np.asarray(t, dtype=float)).slope
    )


def autocorrelation_1(y):
    y = np.asarray(y, dtype=float)
    rang = np.arange(len(y), dtype=float)
    sans = y - pente_sen(y, rang) * rang
    return float(np.corrcoef(sans[:-1], sans[1:])[0, 1])


def blanchiment_simple(y):
    y = np.asarray(y, dtype=float)
    r1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    return y[1:] - r1 * y[:-1], r1


def tfpw(y):
    y = np.asarray(y, dtype=float)
    n = len(y)
    rang = np.arange(n, dtype=float)
    pente = pente_sen(y, rang)
    sans_tendance = y - pente * rang
    r1 = float(np.corrcoef(sans_tendance[:-1], sans_tendance[1:])[0, 1])
    blanchie = sans_tendance[1:] - r1 * sans_tendance[:-1]
    return blanchie + pente * np.arange(1, n, dtype=float), r1


TESTS = {
    "ordinaire": lambda y: (np.asarray(y, dtype=float), None),
    "PW": blanchiment_simple,
    "TFPW": tfpw,
}


def desaisonnaliser(y, mois):
    y = np.asarray(y, dtype=float)
    mois = np.asarray(mois)
    sortie = np.empty_like(y)
    for m in np.unique(mois):
        ou = mois == m
        sortie[ou] = y[ou] - y[ou].mean()
    return sortie


def ar1(n, phi, sigma=1.0, *, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    e = rng.normal(scale=sigma * np.sqrt(max(1.0 - phi * phi, 1e-12)), size=n)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + e[i]
    return s


def taux_de_rejet(
    n, phi, pente=0.0, sigma=1.0, tirages=TIRAGES_H0, graine=GRAINE_H0, alpha=ALPHA
):
    rng = np.random.default_rng(graine)
    rangs = np.arange(n, dtype=float)
    rejets = {nom: 0 for nom in TESTS}
    for _ in range(tirages):
        serie = ar1(n, phi, sigma, rng=rng) + pente * rangs
        for nom, procedure in TESTS.items():
            if mann_kendall(procedure(serie)[0])[3] < alpha:
                rejets[nom] += 1
    return {nom: v / tirages for nom, v in rejets.items()}


def plancher(n, phi, tirages=TIRAGES_H0, graine=GRAINE_H0, alpha=ALPHA):
    return taux_de_rejet(
        n, phi, pente=0.0, sigma=1.0, tirages=tirages, graine=graine, alpha=alpha
    )


def six_chemins(serie, mois, annees, tirages=TIRAGES_H0):
    series = {
        "brute": np.asarray(serie, dtype=float),
        "anomalie": desaisonnaliser(serie, mois),
    }
    annees = np.asarray(annees, dtype=float)
    lignes = []
    for nom_serie, y in series.items():
        phi = autocorrelation_1(y)
        taux = taux_de_rejet(len(y), phi, tirages=tirages)
        pente_serie = pente_sen(y, annees) * 10.0
        rangs = np.arange(len(y), dtype=float)
        pente_par_pas = pente_sen(y, rangs)
        sigma_residu = float(np.std(y - pente_par_pas * rangs))
        puissance = taux_de_rejet(
            len(y),
            phi,
            pente=pente_par_pas,
            sigma=sigma_residu,
            tirages=tirages,
            graine=GRAINE_H0 + 1,
        )
        for nom_test, procedure in TESTS.items():
            z, r1 = procedure(y)
            t = annees[len(annees) - len(z) :]
            s, tau, stat_z, p, var_s = mann_kendall(z)
            lignes.append(
                {
                    "serie": nom_serie,
                    "test": nom_test,
                    "n_mois": len(z),
                    "tau": tau,
                    "z": stat_z,
                    "p": p,
                    "pente_serie_decennie": pente_serie,
                    "pente_testee_decennie": pente_sen(z, t) * 10.0,
                    "r1": r1,
                    "phi_calibration": phi,
                    "rejet_sous_h0": taux[nom_test],
                    "puissance": puissance[nom_test],
                    "var_s": var_s,
                }
            )
    return lignes


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


def produire(chemin_netcdf, tirages=TIRAGES_H0):
    chemin = Path(chemin_netcdf)
    if not chemin.exists():
        raise SystemExit(
            f"source not found: {chemin}\nGive the TWO-SAT NetCDF path with --netcdf."
        )
    valeurs, mois, annees, emprise, temps = serie_mensuelle(chemin)

    lignes = six_chemins(valeurs, mois, annees, tirages=tirages)

    corps = []
    for L in lignes:
        seuil = 1.96 / np.sqrt(L["n_mois"])
        corps.append(
            {
                "serie": L["serie"],
                "test": L["test"],
                "n_mois": L["n_mois"],
                "tau": f"{L['tau']:.4f}",
                "z": f"{L['z']:.4f}",
                "p": f"{L['p']:.6f}",
                "pente_serie_cm2_s2_par_decennie": f"{L['pente_serie_decennie']:+.4f}",
                "pente_testee_cm2_s2_par_decennie": f"{L['pente_testee_decennie']:+.4f}",
                "r1": "" if L["r1"] is None else f"{L['r1']:.4f}",
                "seuil_r1_5pct": f"{seuil:.4f}",
                "phi_calibration": f"{L['phi_calibration']:.4f}",
                "rejet_sous_h0": f"{L['rejet_sous_h0']:.4f}",
                "puissance": f"{L['puissance']:.4f}",
            }
        )

    sauver_table(
        pd.DataFrame(corps),
        "ch03_tab08",
        "tendance eke six chemins et leur plancher",
        "src/ch03_altimetrie/ch03_tab08_tendance_eke.py",
        chemin.name,
        "3.4",
        domaine="bassin-algerien-1993-2023",
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
                "eke_moyenne_cm2_s2": float(valeurs.mean()),
                "eke_ecart_type_cm2_s2": float(valeurs.std()),
                "tirages_simulation": tirages,
                "graine_plancher": GRAINE_H0,
                "graine_puissance": GRAINE_H0 + 1,
                "note": "test de significativite de r1 non applique ; le TFPW est publie mais ne porte aucune conclusion",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return lignes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    ap.add_argument("--tirages", type=int, default=TIRAGES_H0)
    a = ap.parse_args()
    produire(a.netcdf, tirages=a.tirages)
