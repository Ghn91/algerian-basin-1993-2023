"""Mean fields of the TWO-SAT product and the longitudinal profile of mean EKE.

Usage: python ch03_tab10_champs_moyens.py [--netcdf PATH]
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
from ch03_tab08_tendance_eke import NETCDF_DEFAUT, CM2_PAR_M2

ARCHIVE = RACINE / "docs" / "ch03_champs.json"

BESOINS = {
    "topographie_dynamique": (("adt",), "m"),
    "energie_turbulente": (("ugosa", "vgosa"), "cm2_s-2"),
    "ecart_type_niveau": (("sla",), "cm"),
    "vitesse_moyenne": (("ugos", "vgos"), "m_s-1"),
}
AIRES_ECARTEES = ()

QUANTILES = (0.02, 0.25, 0.50, 0.75, 0.98)
PAS_PROFIL = 0.5
MINIMUM_PAS_VALIDES = 2


def variables_manquantes(presentes, besoins=BESOINS):
    presentes = set(presentes)
    manque = {}
    for champ, (variables, _) in besoins.items():
        absentes = [v for v in variables if v not in presentes]
        if absentes:
            manque[champ] = absentes
    return manque


def ecart_type_temporel(champ, minimum=MINIMUM_PAS_VALIDES):
    import warnings

    valides = champ.notnull().sum("time")
    vides = int((valides == 0).sum())
    insuffisantes = int((valides < minimum).sum())
    if insuffisantes != vides:
        raise SystemExit(
            f"{insuffisantes - vides} cell(s) carry between 1 and {minimum - 1} valid time steps; stopping rather than mixing samples."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return champ.std("time"), vides


def resumer(champ2d, longitudes, latitudes, quantiles=QUANTILES):
    valeurs = np.asarray(champ2d, dtype=float)
    plat = valeurs.ravel()
    fini = np.isfinite(plat)
    if not fini.any():
        raise SystemExit("field entirely missing: nothing to summarise")
    indice = int(np.nanargmax(valeurs))
    ligne, colonne = np.unravel_index(indice, valeurs.shape)
    sortie = {
        "mailles": int(fini.sum()),
        "mailles_absentes": int((~fini).sum()),
        "minimum": float(np.nanmin(valeurs)),
        "maximum": float(np.nanmax(valeurs)),
        "moyenne_spatiale": float(np.nanmean(valeurs)),
        "longitude_du_maximum": float(np.asarray(longitudes)[colonne]),
        "latitude_du_maximum": float(np.asarray(latitudes)[ligne]),
    }
    for q in quantiles:
        sortie[f"quantile_{int(q * 100):02d}"] = float(np.nanquantile(plat, q))
    return sortie


def profil_longitudinal(champ2d, longitudes, latitudes=None, pas=PAS_PROFIL):
    valeurs = np.asarray(champ2d, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    lat = None if latitudes is None else np.asarray(latitudes, dtype=float)
    depart = np.floor(lon.min() / pas) * pas
    arrivee = np.ceil(lon.max() / pas) * pas
    bandes = []
    borne = depart
    while borne < arrivee - 1e-9:
        haut = borne + pas
        derniere = haut >= arrivee - 1e-9
        dedans = (lon >= borne) & ((lon <= haut) if derniere else (lon < haut))
        if dedans.any():
            tranche = valeurs[:, dedans]
            fini = np.isfinite(tranche)
            bande = {
                "longitude_min": round(float(borne), 4),
                "longitude_max": round(float(haut), 4),
                "colonnes": int(dedans.sum()),
                "mailles_valides": int(fini.sum()),
                "moyenne": float(np.nanmean(tranche)),
                "maximum": float(np.nanmax(tranche)),
            }
            if lat is not None and fini.any():
                ligne = int(
                    np.unravel_index(int(np.nanargmax(tranche)), tranche.shape)[0]
                )
                bande["latitude_du_maximum"] = float(lat[ligne])
            bandes.append(bande)
        borne = haut
    return bandes


def produire(chemin_netcdf):
    chemin = Path(chemin_netcdf)
    if not chemin.exists():
        raise SystemExit(
            f"source not found: {chemin}\nGive the NetCDF path with --netcdf."
        )
    import xarray as xr

    ds = xr.open_dataset(chemin)
    try:
        manque = variables_manquantes(ds.data_vars)
        if manque:
            detail = " · ".join(
                f"{c} demande {', '.join(v)}" for c, v in manque.items()
            )
            raise SystemExit(
                f"variables missing from the file: {detail}\npresent: {', '.join(sorted(ds.data_vars))}"
            )

        lon = ds["longitude"].values
        lat = ds["latitude"].values

        ecart_sla, mailles_vides = ecart_type_temporel(ds["sla"])
        champs = {
            "topographie_dynamique": ds["adt"].mean("time").values,
            "energie_turbulente": (
                0.5 * (ds["ugosa"] ** 2 + ds["vgosa"] ** 2) * CM2_PAR_M2
            )
            .mean("time")
            .values,
            "ecart_type_niveau": (ecart_sla * 100.0).values,
            "vitesse_moyenne": np.hypot(
                ds["ugos"].mean("time").values, ds["vgos"].mean("time").values
            ),
        }
        premier, dernier = (
            str(ds["time"].values[0])[:10],
            str(ds["time"].values[-1])[:10],
        )
    finally:
        ds.close()

    lignes, archive = [], {}
    for champ, valeurs in champs.items():
        unite = BESOINS[champ][1]
        r = resumer(valeurs, lon, lat)
        lignes.append(
            {
                "champ": champ,
                "unite": unite,
                "mailles": r["mailles"],
                "mailles_absentes": r["mailles_absentes"],
                "minimum": f"{r['minimum']:.4f}",
                "quantile_02": f"{r['quantile_02']:.4f}",
                "quantile_25": f"{r['quantile_25']:.4f}",
                "mediane": f"{r['quantile_50']:.4f}",
                "quantile_75": f"{r['quantile_75']:.4f}",
                "quantile_98": f"{r['quantile_98']:.4f}",
                "maximum": f"{r['maximum']:.4f}",
                "moyenne_spatiale": f"{r['moyenne_spatiale']:.4f}",
                "longitude_du_maximum": f"{r['longitude_du_maximum']:.4f}",
                "latitude_du_maximum": f"{r['latitude_du_maximum']:.4f}",
            }
        )
        archive[champ] = dict(r, unite=unite, variables=list(BESOINS[champ][0]))

    sauver_table(
        pd.DataFrame(lignes),
        "ch03_tab10",
        "champs moyens de la circulation du bassin",
        "src/ch03_altimetrie/ch03_tab10_champs_moyens.py",
        chemin.name,
        "3.3",
        domaine="bassin-algerien-1993-2023",
    )

    bandes = profil_longitudinal(champs["energie_turbulente"], lon, lat)
    sauver_table(
        pd.DataFrame(
            [
                {
                    "longitude_min": f"{b['longitude_min']:.2f}",
                    "longitude_max": f"{b['longitude_max']:.2f}",
                    "colonnes": b["colonnes"],
                    "mailles_valides": b["mailles_valides"],
                    "energie_moyenne_cm2_s-2": f"{b['moyenne']:.4f}",
                    "energie_maximale_cm2_s-2": f"{b['maximum']:.4f}",
                    "latitude_du_maximum": f"{b.get('latitude_du_maximum', float('nan')):.4f}",
                }
                for b in bandes
            ]
        ),
        "ch03_tab11",
        "profil longitudinal de l energie turbulente",
        "src/ch03_altimetrie/ch03_tab10_champs_moyens.py",
        chemin.name,
        "3.3",
        domaine="bassin-algerien-1993-2023",
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "fichier": chemin.name,
                "premier_jour": premier,
                "dernier_jour": dernier,
                "pas_du_profil_degre": PAS_PROFIL,
                "mailles_sans_aucun_pas_valide": mailles_vides,
                "minimum_pas_valides_declare": MINIMUM_PAS_VALIDES,
                "champs": archive,
                "note": "produit a constellation STABLE — voir le § 8 ter b de la "
                "conception ; les valeurs ne sont pas comparables a celles du "
                "travail d'origine, qui emploie l'autre produit",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes, bandes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    a = ap.parse_args()
    produire(a.netcdf)
