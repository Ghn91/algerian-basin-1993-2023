"""Cross-correlation of the WeMO index with two kinetic energies (EKE of the
velocity anomalies; energy of the total flow), lags -24 to +24 months, with
the effective sample size and a Bonferroni threshold over the lags.

Usage: python ch03_tab20_correlation_wemo.py [--netcdf PATH]
"""

import argparse
import csv
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
ABSOLUES = ("ugos", "vgos")

PUBLIE = {"r_zero": -0.176, "retard": 1, "r_max": -0.220, "n": 372}
TOLERANCE_PUBLIE = 0.002

WEMO_THESE = RACINE / "tables" / "ch05" / "ch05_tab23_indices_reconstruits_era5.csv"
WEMO_PUBLIEE = (
    RACINE
    / "tables"
    / "ch03"
    / "ch03_tab18_serie_wemo_publiee_ub_barcelone_transportee.csv"
)
WEMO_ORIGINE = Path("revision/origine_2026-03/wemo_index_era5_1993_2023.csv")
DERIVE_EKE = RACINE / "data" / "interim" / "eke_mensuelle_twosat.csv"
ARCHIVE = RACINE / "docs" / "ch03_correlation_wemo.json"

CM2_PAR_M2 = 1e4
RETARD_MAX = 24
ALPHA = 0.05


def correlation_croisee(indice, ocean, retard_max=RETARD_MAX):
    a = np.asarray(indice, dtype=float)
    b = np.asarray(ocean, dtype=float)
    if len(a) != len(b):
        raise SystemExit("the two series do not have the same length")
    lignes = []
    for k in range(-retard_max, retard_max + 1):
        if k > 0:
            x, y = a[:-k], b[k:]
        elif k < 0:
            x, y = a[-k:], b[:k]
        else:
            x, y = a, b
        if len(x) < 24:
            continue
        r, p = stats.pearsonr(x, y)
        lignes.append(
            {"retard_mois": k, "r": float(r), "p_brut": float(p), "n": len(x)}
        )
    return lignes


def retard_du_maximum(lignes):
    meilleur = max(lignes, key=lambda L: abs(L["r"]))
    return meilleur["retard_mois"], meilleur


def degres_de_liberte_effectifs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return float(n), float("nan"), float("nan")
    r1x = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1y = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    produit = r1x * r1y
    if produit >= 1.0:
        return 0.0, r1x, r1y
    n_eff = n * (1.0 - produit) / (1.0 + produit)
    return float(min(n_eff, n)), r1x, r1y


def seuil_correlation(n_eff, alpha=ALPHA, tests=1):
    if n_eff < 4:
        return 1.0
    z = stats.norm.ppf(1.0 - alpha / (2.0 * tests))
    return float(np.tanh(z / np.sqrt(n_eff - 3.0)))


def retards_significatifs(lignes, n_eff, alpha=ALPHA, tests=1):
    seuil = seuil_correlation(n_eff, alpha, tests)
    retenus = sorted(L["retard_mois"] for L in lignes if abs(L["r"]) >= seuil)
    if not retenus:
        return {
            "seuil": seuil,
            "combien": 0,
            "de": None,
            "a": None,
            "contigus": None,
            "libelle": "aucun",
        }
    contigus = (retenus[-1] - retenus[0] + 1) == len(retenus)
    libelle = (
        f"{retenus[0]:+d} a {retenus[-1]:+d}"
        if contigus
        else f"{len(retenus)} retards disperses entre {retenus[0]:+d} "
        f"et {retenus[-1]:+d}"
    )
    return {
        "seuil": seuil,
        "combien": len(retenus),
        "de": retenus[0],
        "a": retenus[-1],
        "contigus": "oui" if contigus else "non",
        "libelle": libelle,
    }


def energies_mensuelles(chemin):
    import xarray as xr

    ds = xr.open_dataset(chemin)
    sorties = {}
    try:
        for jeu, (cu, cv) in (("turbulente", ANOMALIES), ("totale", ABSOLUES)):
            absentes = [v for v in (cu, cv) if v not in ds.data_vars]
            if absentes:
                raise SystemExit(
                    f"the file lacks {', '.join(absentes)}.\npresent: {', '.join(sorted(ds.data_vars))}\nBoth the velocity anomalies and the absolute velocities are required."
                )
            for ordre in ("quotidienne", "mensuelle"):
                if ordre == "quotidienne":
                    champ = (
                        (0.5 * (ds[cu] ** 2 + ds[cv] ** 2) * CM2_PAR_M2)
                        .resample(time="MS")
                        .mean()
                    )
                else:
                    u = ds[cu].resample(time="MS").mean()
                    v = ds[cv].resample(time="MS").mean()
                    champ = 0.5 * (u**2 + v**2) * CM2_PAR_M2
                spatial = champ.mean(dim=[d for d in champ.dims if d != "time"])
                sorties[f"{jeu}_{ordre}"] = dict(
                    zip(
                        [str(t)[:7] for t in champ["time"].values],
                        np.asarray(spatial.values, dtype=float),
                    )
                )
    finally:
        ds.close()
    return sorties


DU_MANUSCRIT = "turbulente_quotidienne"
DE_LA_FIGURE_8 = "totale_mensuelle"


def lire_derive(chemin):
    p = Path(chemin)
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    colonnes = {
        "turbulente_quotidienne": "eke_cm2_s2",
        "turbulente_mensuelle": "eke_vitesses_mensuelles_cm2_s2",
        "totale_quotidienne": "energie_totale_cm2_s2",
        "totale_mensuelle": "energie_totale_vitesses_mensuelles_cm2_s2",
    }
    if not lignes or any(c not in lignes[0] for c in colonnes.values()):
        return None
    return {
        cle: {r["mois"]: float(r[col]) for r in lignes} for cle, col in colonnes.items()
    }


def ecrire_derive(chemin, series):
    colonnes = [
        ("turbulente_quotidienne", "eke_cm2_s2"),
        ("turbulente_mensuelle", "eke_vitesses_mensuelles_cm2_s2"),
        ("totale_quotidienne", "energie_totale_cm2_s2"),
        ("totale_mensuelle", "energie_totale_vitesses_mensuelles_cm2_s2"),
    ]
    p = Path(chemin)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mois"] + [c for _, c in colonnes])
        for mois in sorted(series[DU_MANUSCRIT]):
            w.writerow([mois] + [f"{series[cle][mois]:.10f}" for cle, _ in colonnes])


def lire_wemo_these(chemin, an_max=2023):
    with open(chemin, newline="", encoding="utf-8") as f:
        return {
            f'{int(r["annee"]):04d}-{int(r["mois"]):02d}': float(r["wemo_era5"])
            for r in csv.DictReader(f)
            if int(r["annee"]) <= an_max
        }


def lire_wemo_publiee(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        return {r["mois"]: float(r["wemo_ub_publiee"]) for r in csv.DictReader(f)}


def lire_wemo_origine(chemin=WEMO_ORIGINE):
    p = Path(chemin)
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        return {r["time"]: float(r["wemo_era5"]) for r in csv.DictReader(f)}


def alignees(a, b):
    communs = sorted(set(a) & set(b))
    return (
        communs,
        np.array([a[m] for m in communs]),
        np.array([b[m] for m in communs]),
    )


def desaisonnaliser(serie):
    par_mois = {}
    for cle, v in serie.items():
        par_mois.setdefault(cle[5:7], []).append(v)
    moyennes = {m: float(np.mean(v)) for m, v in par_mois.items()}
    return {cle: v - moyennes[cle[5:7]] for cle, v in serie.items()}


def empreinte(chemin, bloc=1 << 20):
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for morceau in iter(lambda: f.read(bloc), b""):
            h.update(morceau)
    return h.hexdigest()


def produire(chemin_netcdf=NETCDF_DEFAUT):
    for p, quoi in (
        (WEMO_THESE, "la WeMO de la these"),
        (WEMO_PUBLIEE, "la serie UB transportee"),
    ):
        if not Path(p).exists():
            raise SystemExit(f"source not found: {p}\n({quoi})")

    chemin = Path(chemin_netcdf)
    archivee = lire_derive(DERIVE_EKE)
    if chemin.exists():
        energies = energies_mensuelles(chemin)
        if archivee is not None:
            for cle in energies:
                communs = sorted(set(energies[cle]) & set(archivee[cle]))
                ecart = max(abs(energies[cle][m] - archivee[cle][m]) for m in communs)
                if ecart > 1e-6:
                    raise SystemExit(
                        f"  the archive and the computation differ by {ecart:.3e}; delete the archive and rerun."
                    )
        ecrire_derive(DERIVE_EKE, energies)
    elif archivee is not None:
        energies = archivee
    else:
        raise SystemExit(
            f"neither the NetCDF ({chemin}) nor the archive ({DERIVE_EKE}).\nGive the NetCDF path with --netcdf to produce the archive."
        )

    com = sorted(set(energies[DU_MANUSCRIT]) & set(energies[DE_LA_FIGURE_8]))

    def r_entre(a, b):
        return float(
            np.corrcoef([energies[a][m] for m in com], [energies[b][m] for m in com])[
                0, 1
            ]
        )

    r_energies = r_entre(DU_MANUSCRIT, DE_LA_FIGURE_8)
    r_jeu = r_entre("turbulente_quotidienne", "totale_quotidienne")
    r_ordre = r_entre("turbulente_quotidienne", "turbulente_mensuelle")

    wemo = lire_wemo_these(WEMO_THESE)
    publiee = lire_wemo_publiee(WEMO_PUBLIEE)
    origine = lire_wemo_origine()

    indices = {
        "reconstruction de la these": desaisonnaliser(wemo),
        "serie publiee UB": desaisonnaliser(publiee),
    }
    restreinte = {m: v for m, v in wemo.items() if m in publiee}
    indices["reconstruction de la these, bornee aux mois de UB"] = desaisonnaliser(
        restreinte
    )
    if origine:
        indices["reconstruction d'origine (signe inverse)"] = desaisonnaliser(origine)
    energies_a = {k: desaisonnaliser(v) for k, v in energies.items()}

    reproduction = None
    if origine:
        mois_o, xo, yo = alignees(
            indices["reconstruction d'origine (signe " "inverse)"],
            energies_a[DE_LA_FIGURE_8],
        )
        bal_o = correlation_croisee(xo, yo)
        r0 = next(L["r"] for L in bal_o if L["retard_mois"] == 0)
        ko, meilleur_o = retard_du_maximum(bal_o)
        ecarts = (abs(r0 - PUBLIE["r_zero"]), abs(meilleur_o["r"] - PUBLIE["r_max"]))
        reproduction = {
            "n": len(mois_o),
            "r_zero": r0,
            "retard": ko,
            "r_max": meilleur_o["r"],
            "ecart_r_zero": ecarts[0],
            "ecart_r_max": ecarts[1],
        }
        if not (max(ecarts) <= TOLERANCE_PUBLIE and ko == PUBLIE["retard"]):
            raise SystemExit(
                f"\n  REPRODUCTION FAILED (differences {ecarts[0]:.4f} and {ecarts[1]:.4f}); nothing written."
            )

    lignes_balayage, lignes_verdict = [], []
    for (nom_e, eke_a), (nom, indice) in [
        (e, i) for e in energies_a.items() for i in indices.items()
    ]:
        mois, x, y = alignees(indice, eke_a)
        n_eff, r1x, r1y = degres_de_liberte_effectifs(x, y)
        balayage = correlation_croisee(x, y)
        k_opt, meilleur = retard_du_maximum(balayage)
        tests = len(balayage)
        nom_l = retards_significatifs(balayage, n_eff)
        bon = retards_significatifs(balayage, n_eff, tests=tests)
        r_zero = next(L["r"] for L in balayage if L["retard_mois"] == 0)

        survit = abs(meilleur["r"]) >= bon["seuil"]

        for L in balayage:
            lignes_balayage.append({"energie": nom_e, "serie": nom, **L})
        lignes_verdict.append(
            {
                "energie": nom_e,
                "serie": nom,
                "n_mois": len(mois),
                "premier_mois": mois[0],
                "dernier_mois": mois[-1],
                "r1_indice": r1x,
                "r1_eke": r1y,
                "n_effectif": n_eff,
                "retard_du_maximum": k_opt,
                "r_au_maximum": meilleur["r"],
                "r_au_retard_zero": r_zero,
                "seuil_nominal": nom_l["seuil"],
                "retards_significatifs_nominal": nom_l["combien"],
                "etendue_nominale": nom_l["libelle"],
                "retards_contigus_nominal": nom_l["contigus"] or "sans objet",
                "tests_balayes": tests,
                "seuil_bonferroni": bon["seuil"],
                "retards_significatifs_bonferroni": bon["combien"],
                "etendue_bonferroni": bon["libelle"],
                "retards_contigus_bonferroni": bon["contigus"] or "sans objet",
                "pic_survit_a_bonferroni": "oui" if survit else "non",
            }
        )

    SCRIPT = "src/ch03_altimetrie/ch03_tab20_correlation_wemo.py"
    DOM = "bassin-algerien-1993-2023"

    def texte(rows):
        return pd.DataFrame(
            [
                {c: (f"{v:.6f}" if isinstance(v, float) else v) for c, v in L.items()}
                for L in rows
            ]
        )

    sauver_table(
        texte(lignes_balayage),
        "ch03_tab20",
        "balayage des retards de la correlation wemo energie",
        SCRIPT,
        chemin.name if chemin.exists() else DERIVE_EKE.name,
        "3.7",
        domaine=DOM,
    )
    sauver_table(
        texte(lignes_verdict),
        "ch03_tab21",
        "verdict de la correlation wemo energie et son effectif",
        SCRIPT,
        chemin.name if chemin.exists() else DERIVE_EKE.name,
        "3.7",
        domaine=DOM,
    )

    lignes_plan = []
    for cle in (
        "turbulente_quotidienne",
        "turbulente_mensuelle",
        "totale_quotidienne",
        "totale_mensuelle",
    ):
        lignes_plan.append(
            {
                "serie": cle,
                "jeu_de_composantes": (
                    "anomalies ugosa vgosa"
                    if cle.startswith("turbulente")
                    else "absolues ugos vgos"
                ),
                "ordre_des_operations": (
                    "carre quotidien puis moyenne"
                    if cle.endswith("quotidienne")
                    else "moyenne mensuelle des vitesses " "puis carre"
                ),
                "statut": (
                    "definie a l equation 1 du manuscrit"
                    if cle == DU_MANUSCRIT
                    else (
                        "calculee par le script de sa figure 8"
                        if cle == DE_LA_FIGURE_8
                        else "croisee"
                    )
                ),
                "moyenne_cm2_s2": float(np.mean(list(energies[cle].values()))),
                "r_contre_la_serie_definie": r_entre(DU_MANUSCRIT, cle),
            }
        )
    if reproduction:
        lignes_plan.append(
            {
                "serie": "reproduction du chiffre publie",
                "jeu_de_composantes": "absolues ugos vgos",
                "ordre_des_operations": "moyenne mensuelle des vitesses puis carre",
                "statut": "recette de la figure 8 de la version soumise",
                "moyenne_cm2_s2": float("nan"),
                "r_contre_la_serie_definie": float("nan"),
                "r_zero_publie": PUBLIE["r_zero"],
                "r_zero_reproduit": reproduction["r_zero"],
                "ecart_au_r_zero_publie": reproduction["ecart_r_zero"],
                "r_max_publie": PUBLIE["r_max"],
                "r_max_reproduit": reproduction["r_max"],
                "ecart_au_r_max_publie": reproduction["ecart_r_max"],
                "retard_publie": PUBLIE["retard"],
                "retard_reproduit": reproduction["retard"],
                "n_mois_publie": PUBLIE["n"],
            }
        )
    lignes_plancher = []
    for etiquette, (phi_i, phi_e, taille) in (
        ("regime de la reconstruction sur 372 mois", (0.013, 0.692, 372)),
        ("regime de la serie publiee sur 252 mois", (0.172, 0.724, 252)),
    ):
        nominal = bonf = 0
        tirages = 200
        for g in range(tirages):
            rg = np.random.default_rng(7000 + g)

            def ar1(phi, n, r):
                s = np.zeros(n)
                inn = np.sqrt(1.0 - phi**2)
                for i in range(1, n):
                    s[i] = phi * s[i - 1] + inn * r.normal()
                return s

            a = ar1(phi_i, taille, rg)
            b = ar1(phi_e, taille, rg)
            bal = correlation_croisee(a, b)
            _, pic = retard_du_maximum(bal)
            ne = degres_de_liberte_effectifs(a, b)[0]
            if abs(pic["r"]) >= seuil_correlation(ne):
                nominal += 1
            if abs(pic["r"]) >= seuil_correlation(ne, tests=len(bal)):
                bonf += 1
        lignes_plancher.append(
            {
                "regime": etiquette,
                "r1_indice_simule": phi_i,
                "r1_energie_simule": phi_e,
                "n_mois": taille,
                "tirages": tirages,
                "risque_annonce_pourcent": 100 * ALPHA,
                "plancher_au_risque_nominal_pourcent": 100 * nominal / tirages,
                "plancher_avec_bonferroni_pourcent": 100 * bonf / tirages,
            }
        )
    sauver_table(
        texte(lignes_plancher),
        "ch03_tab25",
        "plancher du test de correlation croisee sur series " "independantes",
        SCRIPT,
        "series simulees",
        "3.7",
        domaine=DOM,
    )

    sauver_table(
        texte(lignes_plan),
        "ch03_tab24",
        "plan factoriel des definitions d energie et reproduction du " "chiffre publie",
        SCRIPT,
        chemin.name if chemin.exists() else DERIVE_EKE.name,
        "3.7",
        domaine=DOM,
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "convention_du_retard": "r(k) = corr(indice[t], eke[t+k]) ; "
                "k > 0 signifie que l'indice PRECEDE l'EKE",
                "convention_de_l_indice": "WeMO = San Fernando moins Padoue "
                "(definition canonique)",
                "plan_factoriel": {
                    "definie_au_manuscrit": DU_MANUSCRIT,
                    "calculee_par_sa_figure_8": DE_LA_FIGURE_8,
                    "jeu": "anomalies ugosa/vgosa (equation 1) ou absolues ugos/vgos",
                    "ordre": "carre au pas quotidien puis moyenne (fig03, § 3.4 a 3.6),"
                    " ou moyenne mensuelle des vitesses puis carre (fig08)",
                    "moyennes_cm2_s2": {
                        k: float(np.mean(list(v.values()))) for k, v in energies.items()
                    },
                    "correlation_manuscrit_contre_figure_8": r_energies,
                    "correlation_ecart_de_jeu_seul": r_jeu,
                    "correlation_ecart_d_ordre_seul": r_ordre,
                },
                "reproduction_du_chiffre_publie": reproduction,
                "publie_au_manuscrit": PUBLIE,
                "derive_eke": {
                    "fichier": str(DERIVE_EKE.relative_to(RACINE)).replace("\\", "/"),
                    "mois": len(energies[DU_MANUSCRIT]),
                    "sha256": empreinte(DERIVE_EKE),
                },
                "netcdf": (
                    {"fichier": chemin.name, "sha256": empreinte(chemin)}
                    if chemin.exists()
                    else "absent — repli sur le derive"
                ),
                "retard_max_balaye": RETARD_MAX,
                "risque": ALPHA,
                "reference_n_effectif": "Bretherton et al. (1999), eq. 31",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes_verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    a = ap.parse_args()
    produire(a.netcdf)
