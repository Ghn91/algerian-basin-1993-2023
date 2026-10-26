"""Moving-block bootstrap interval of the Sen slope of the EKE anomaly, for ten
block lengths, with its coverage measured on simulated series and corrected.

Usage: python ch03_tab09_incertitude_pente.py [--netcdf PATH] [--tirages N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))
sys.path.append(str(Path(__file__).resolve().parent))
import pandas as pd
from provenance import sauver_table
from ch03_tab08_tendance_eke import (
    NETCDF_DEFAUT,
    ar1,
    autocorrelation_1,
    desaisonnaliser,
    pente_sen,
    serie_mensuelle,
)

ARCHIVE = RACINE / "docs" / "ch03_incertitude.json"
BLOCS = (1, 3, 5, 8, 12, 18, 24, 36, 48, 60)
TIRAGES_IC = 600
TIRAGES_COUVERTURE = 120
NIVEAU = 0.95
GRAINE_IC = 2609
GRAINE_COUVERTURE = 2610
GRAINE_INFLATION = 2611
LARGEURS_FACTEUR = 100


class PenteRapide:

    def __init__(self, t):
        t = np.asarray(t, dtype=float)
        i, j = np.triu_indices(len(t), k=1)
        dt = t[j] - t[i]
        garde = dt != 0
        self.i, self.j, self.dt = i[garde], j[garde], dt[garde]
        self.n = len(t)

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        if len(y) != self.n:
            raise ValueError(
                f"series of {len(y)} points for a grid of {self.n}: a PenteRapide object is valid for its own time grid only"
            )
        return float(np.median((y[self.j] - y[self.i]) / self.dt))


def blocs_mobiles(residu, longueur, rng):
    n = len(residu)
    if longueur >= n:
        longueur = n - 1
    departs = rng.integers(0, n - longueur + 1, size=int(np.ceil(n / longueur)))
    return np.concatenate([residu[d : d + longueur] for d in departs])[:n]


def intervalle(y, t, longueur, tirages=TIRAGES_IC, niveau=NIVEAU, graine=GRAINE_IC):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    b = pente_sen(y, t)
    rapide = PenteRapide(t)
    residu = y - b * t
    rng = np.random.default_rng(graine)
    pentes = np.empty(tirages)
    for i in range(tirages):
        pentes[i] = rapide(b * t + blocs_mobiles(residu, longueur, rng))
    alpha = (1.0 - niveau) / 2.0
    bas, haut = np.quantile(pentes, [alpha, 1.0 - alpha])
    return b, float(bas), float(haut), float(np.std(pentes))


def facteur_inflation(
    n,
    phi,
    sigma,
    pente_vraie,
    longueur,
    series=200,
    tirages_ic=150,
    largeurs=LARGEURS_FACTEUR,
    niveau=NIVEAU,
    graine=GRAINE_INFLATION,
):
    rng = np.random.default_rng(graine)
    t = np.arange(n, dtype=float)
    rapide = PenteRapide(t)
    alpha = (1.0 - niveau) / 2.0
    vraies = np.fromiter(
        (rapide(sigma * ar1(n, phi, rng) + pente_vraie * t) for _ in range(series)),
        dtype=float,
        count=series,
    )
    bas, haut = np.quantile(vraies, [alpha, 1.0 - alpha])
    reelle = haut - bas
    mesures = []
    for _ in range(largeurs):
        y = sigma * ar1(n, phi, rng) + pente_vraie * t
        b = rapide(y)
        residu = y - b * t
        pentes = np.fromiter(
            (
                rapide(b * t + blocs_mobiles(residu, longueur, rng))
                for _ in range(tirages_ic)
            ),
            dtype=float,
            count=tirages_ic,
        )
        lo, hi = np.quantile(pentes, [alpha, 1.0 - alpha])
        mesures.append(hi - lo)
    mediane = float(np.median(mesures))
    return reelle / mediane if mediane > 0 else float("inf")


def couverture(
    n,
    phi,
    sigma,
    pente_vraie,
    longueur,
    facteur=1.0,
    tirages=TIRAGES_COUVERTURE,
    niveau=NIVEAU,
    graine=GRAINE_COUVERTURE,
    tirages_ic=120,
):
    rng = np.random.default_rng(graine)
    t = np.arange(n, dtype=float)
    rapide = PenteRapide(t)
    alpha = (1.0 - niveau) / 2.0
    brute = corrigee = 0
    for _ in range(tirages):
        y = sigma * ar1(n, phi, rng) + pente_vraie * t
        b = rapide(y)
        residu = y - b * t
        pentes = np.fromiter(
            (
                rapide(b * t + blocs_mobiles(residu, longueur, rng))
                for _ in range(tirages_ic)
            ),
            dtype=float,
            count=tirages_ic,
        )
        bas, haut = np.quantile(pentes, [alpha, 1.0 - alpha])
        brute += int(bas <= pente_vraie <= haut)
        centre, demi = (bas + haut) / 2.0, (haut - bas) / 2.0 * facteur
        corrigee += int(centre - demi <= pente_vraie <= centre + demi)
    return brute / tirages, corrigee / tirages


def temps_de_decorrelation(r1):
    return (1.0 + r1) / (1.0 - r1) if r1 < 1.0 else float("inf")


def bloc_retenu(couvertures, facteurs, nominal=NIVEAU, marge=0.02):
    couvrants = [L for L in sorted(couvertures) if couvertures[L] >= nominal - marge]
    if not couvrants:
        return None
    return min(couvrants, key=lambda L: (round(facteurs[L], 6), L))


def produire(
    chemin_netcdf,
    blocs=BLOCS,
    tirages=TIRAGES_IC,
    tirages_couverture=TIRAGES_COUVERTURE,
):
    chemin = Path(chemin_netcdf)
    if not chemin.exists():
        raise SystemExit(
            f"source not found: {chemin}\nGive the TWO-SAT NetCDF path with --netcdf."
        )
    valeurs, mois, annees, emprise, temps = serie_mensuelle(chemin)
    anomalie = desaisonnaliser(valeurs, mois)
    n = len(anomalie)
    rangs = np.arange(n, dtype=float)

    phi = autocorrelation_1(anomalie)
    pente_par_pas = pente_sen(anomalie, rangs)
    sigma = float(np.std(anomalie - pente_par_pas * rangs))
    tau_integral = temps_de_decorrelation(phi)
    regle_cubique = n ** (1 / 3)

    lignes, mesures, facteurs = [], {}, {}
    for longueur in blocs:
        b, bas, haut, ecart = intervalle(anomalie, annees, longueur, tirages=tirages)
        k = facteur_inflation(n, phi, sigma, pente_par_pas, longueur)
        cov_brute, cov_corr = couverture(
            n,
            phi,
            sigma,
            pente_par_pas,
            longueur,
            facteur=k,
            tirages=tirages_couverture,
        )
        _, cov_zero = couverture(
            n,
            phi,
            sigma,
            0.0,
            longueur,
            facteur=k,
            tirages=tirages_couverture,
            graine=GRAINE_COUVERTURE + 2,
        )
        fausse_exclusion = 1.0 - cov_zero
        mesures[longueur] = cov_corr
        facteurs[longueur] = k
        b10, bas10, haut10 = b * 10, bas * 10, haut * 10
        centre, demi = (bas10 + haut10) / 2, (haut10 - bas10) / 2 * k
        bas_c, haut_c = centre - demi, centre + demi
        exclut = not (bas_c <= 0.0 <= haut_c)
        lignes.append(
            {
                "bloc_mois": longueur,
                "pente_cm2_s2_par_decennie": f"{b10:+.4f}",
                "facteur_inflation": f"{k:.4f}",
                "borne_basse_brute": f"{bas10:+.4f}",
                "borne_haute_brute": f"{haut10:+.4f}",
                "borne_basse_corrigee": f"{bas_c:+.4f}",
                "borne_haute_corrigee": f"{haut_c:+.4f}",
                "largeur_corrigee": f"{haut_c - bas_c:.4f}",
                "ecart_type_bootstrap": f"{ecart * 10:.4f}",
                "couverture_brute": f"{cov_brute:.4f}",
                "couverture_corrigee": f"{cov_corr:.4f}",
                "fausse_exclusion_de_zero": f"{fausse_exclusion:.4f}",
                "exclut_zero": "oui" if exclut else "non",
            }
        )

    retenu = bloc_retenu(mesures, facteurs)
    if retenu is None:
        verdict = "aucune longueur de bloc n'atteint le taux nominal"
    else:
        r = next(x for x in lignes if x["bloc_mois"] == retenu)
        verdict = (
            f"L = {retenu} mois, couverture corrigee "
            f"{mesures[retenu]:.3f}, intervalle corrige "
            f"[{r['borne_basse_corrigee']} ; "
            f"{r['borne_haute_corrigee']}] cm2 s-2 par decennie, zero "
            f"{'exclu' if r['exclut_zero'] == 'oui' else 'inclus'}"
        )

    sauver_table(
        pd.DataFrame(lignes),
        "ch03_tab09",
        "incertitude de la pente eke par blocs",
        "src/ch03_altimetrie/ch03_tab09_incertitude_pente.py",
        chemin.name,
        "3.4",
        domaine="bassin-algerien-1993-2023",
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "fichier": chemin.name,
                "serie": "anomalie desaisonnalisee",
                "n_mois": n,
                "r1": phi,
                "sigma_residuel_cm2_s2": sigma,
                "reperes_non_decisionnels": {
                    "n_puissance_un_tiers_mois": regle_cubique,
                    "temps_de_decorrelation_mois": tau_integral,
                },
                "niveau": NIVEAU,
                "tirages_intervalle": tirages,
                "tirages_couverture": tirages_couverture,
                "graine_intervalle": GRAINE_IC,
                "graine_couverture": GRAINE_COUVERTURE,
                "longueur_retenue_mois": retenu,
                "verdict": verdict,
                "note": "la longueur de bloc est fixee par le taux de couverture "
                "mesure, jamais par une regle de pouce ; les deux reperes "
                "ci-dessus sont publies pour comparaison et ne decident rien",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes, retenu


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    ap.add_argument("--tirages", type=int, default=TIRAGES_IC)
    ap.add_argument("--couverture", type=int, default=TIRAGES_COUVERTURE)
    a = ap.parse_args()
    produire(a.netcdf, tirages=a.tirages, tirages_couverture=a.couverture)
