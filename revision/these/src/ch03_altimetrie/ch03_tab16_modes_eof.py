"""EOF modes of the SLA anomaly (TWO-SAT), as computed and detrended, and the
zero crossings of the first principal component against a reference.

Usage: python ch03_tab16_modes_eof.py [--netcdf PATH] [--tirages N]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))

import pandas as pd
from provenance import sauver_table

from ch03_altimetrie.ch03_tab12_transition_eke import (
    ar1,
    desaisonnaliser,
    franchissements,
)

NETCDF_DEFAUT = Path(
    "revision/revision_2026/donnees/c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc"
)
VARIABLE = "sla"

ARCHIVE = RACINE / "docs" / "ch03_modes.json"
MODES = 4
FENETRE = 12
TIRAGES = 400
GRAINE = 2806
MOIS_PAR_AN = 12


def poids_latitude(latitudes):
    return np.sqrt(np.cos(np.deg2rad(np.asarray(latitudes, dtype=float))))


def decomposer(matrice, poids=None, modes=MODES):
    m = np.asarray(matrice, dtype=float)
    if poids is not None:
        m = m * np.asarray(poids, dtype=float)[None, :]
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    variance = s**2
    parts = variance / variance.sum()
    modes = min(modes, len(s))
    pcs = u[:, :modes] * s[:modes]
    motifs = vt[:modes, :]
    for i in range(modes):
        extremum = motifs[i][int(np.argmax(np.abs(motifs[i])))]
        if extremum < 0:
            motifs[i] = -motifs[i]
            pcs[:, i] = -pcs[:, i]
    return parts[:modes], pcs, motifs


def retirer_tendance_par_maille(matrice):
    m = np.asarray(matrice, dtype=float)
    t = np.arange(m.shape[0], dtype=float)
    t = t - t.mean()
    pentes = (t[:, None] * m).sum(axis=0) / (t**2).sum()
    return m - t[:, None] * pentes[None, :]


def champ_mensuel(chemin):
    import xarray as xr

    ds = xr.open_dataset(chemin)
    try:
        if VARIABLE not in ds.data_vars:
            ds.close()
            raise SystemExit(
                f"the file lacks `{VARIABLE}`.\npresent: {', '.join(sorted(ds.data_vars))}"
            )
        mensuel = ds[VARIABLE].resample(time="MS").mean()
        temps = mensuel["time"].values
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
        emprise = {
            "longitude_min": float(lon.min()),
            "longitude_max": float(lon.max()),
            "latitude_min": float(lat.min()),
            "latitude_max": float(lat.max()),
            "pas_degre": float(abs(np.diff(lon)[0])),
        }
        cube = np.asarray(
            mensuel.transpose("time", "latitude", "longitude").values, dtype=float
        )
    finally:
        ds.close()
    n_temps = cube.shape[0]
    plat = cube.reshape(n_temps, -1)
    valides = ~np.isnan(plat).any(axis=0)
    partielles = int((np.isnan(plat).any(axis=0) & ~np.isnan(plat).all(axis=0)).sum())
    grille_lat = np.repeat(lat, len(lon))
    mois = np.array([int(str(t)[5:7]) for t in temps])
    retenues = plat[:, valides]
    for j in range(retenues.shape[1]):
        retenues[:, j] = desaisonnaliser(retenues[:, j], mois)
    an0 = int(str(temps[0])[:4])
    annees = an0 + np.arange(n_temps) / MOIS_PAR_AN
    return (
        retenues,
        grille_lat[valides],
        temps,
        annees,
        plat.shape[1],
        partielles,
        emprise,
        valides,
        plat,
        mois,
    )


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
    (
        champ,
        lat,
        temps,
        annees,
        mailles_totales,
        partielles,
        emprise,
        valides,
        plat,
        mois,
    ) = champ_mensuel(chemin)
    if partielles:
        raise SystemExit(
            f"{partielles} partially valid cell(s): two samples would be mixed in one column; stopping."
        )
    residu_centrage = float(np.abs(champ.mean(axis=0)).max())
    if residu_centrage > 1e-9:
        raise SystemExit(
            f"the field is not centred: maximum residual {residu_centrage:.2e}; stopping."
        )

    poids = poids_latitude(lat)
    sans = retirer_tendance_par_maille(champ)

    lignes = []
    for nom, matrice in (("tel quel", champ), ("detendance", sans)):
        parts, pcs, _ = decomposer(matrice, poids)
        lignes.append((nom, parts, pcs))

    rempli = np.nan_to_num(plat, nan=0.0)
    for j in range(rempli.shape[1]):
        rempli[:, j] = desaisonnaliser(rempli[:, j], mois)
    parts_rempli = decomposer(rempli, None)[0]
    parts_sans_poids = decomposer(champ, None)[0]

    pc1 = lignes[0][2][:, 0]
    reel = franchissements(pc1, annees)
    milieu = float((annees[0] + annees[-1]) / 2.0)
    rangs = np.arange(len(pc1), dtype=float)
    pente = float(np.polyfit(rangs, pc1, 1)[0])
    residu = pc1 - pente * rangs
    phi = float(np.corrcoef(residu[:-1], residu[1:])[0, 1])
    sigma = float(np.std(residu))
    rng = np.random.default_rng(GRAINE)
    temoin = {"premier": [], "median": [], "dernier": [], "nombre": []}
    for _ in range(tirages):
        faux = pente * rangs + ar1(len(rangs), phi, sigma, rng=rng)
        f = franchissements(faux - faux.mean(), annees)
        for cle in temoin:
            temoin[cle].append(f[cle])
    loi = {
        c: (
            float(np.nanmedian(v)),
            float(np.nanquantile(v, 0.02)),
            float(np.nanquantile(v, 0.98)),
        )
        for c, v in ((c, np.array(t, dtype=float)) for c, t in temoin.items())
    }
    for cle in ("premier", "median", "dernier", "nombre"):
        m, q2, q98 = loi[cle]
        f = "{:.2f}" if cle != "nombre" else "{:.0f}"

    SCRIPT = "src/ch03_altimetrie/ch03_tab16_modes_eof.py"
    DOM = "bassin-algerien-1993-2023"

    def texte(rows):
        return pd.DataFrame(
            [
                {c: (f"{v:.4f}" if isinstance(v, float) else v) for c, v in L.items()}
                for L in rows
            ]
        )

    corps = []
    for i in range(MODES):
        corps.append(
            {
                "mode": f"EOF{i + 1}",
                "part_tel_quel": float(lignes[0][1][i]),
                "part_detendance": float(lignes[1][1][i]),
                "ecart": float(lignes[1][1][i] - lignes[0][1][i]),
                "sensibilite_remplissage_zero": float(parts_rempli[i]),
                "sensibilite_sans_ponderation": float(parts_sans_poids[i]),
            }
        )
    sauver_table(
        texte(corps),
        "ch03_tab16",
        "parts de variance des modes eof avec et sans tendance",
        SCRIPT,
        chemin.name,
        "3.6",
        domaine=DOM,
    )

    sauver_table(
        texte(
            [
                {
                    "statistique": cle,
                    "pc1_reelle": reel[cle],
                    "temoin_sans_rupture_median": loi[cle][0],
                    "temoin_quantile_02": loi[cle][1],
                    "temoin_quantile_98": loi[cle][2],
                    "milieu_de_la_periode": milieu,
                    "temoin_r1": phi,
                    "tirages": tirages,
                }
                for cle in ("premier", "median", "dernier", "nombre")
            ]
        ),
        "ch03_tab17",
        "franchissement de zero de pc1 et son temoin",
        SCRIPT,
        chemin.name,
        "3.6",
        domaine=DOM,
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "fichier": chemin.name,
                "sha256": empreinte(chemin),
                "octets": chemin.stat().st_size,
                "variable": VARIABLE,
                "emprise_lue": emprise,
                "mailles_totales": int(mailles_totales),
                "mailles_valides": int(champ.shape[1]),
                "mailles_partielles": partielles,
                "premier_mois": str(temps[0])[:7],
                "dernier_mois": str(temps[-1])[:7],
                "milieu_de_la_periode": milieu,
                "ponderation": "racine du cosinus de la latitude",
                "convention_de_signe": "moyenne spatiale du motif rendue positive",
                "fenetre_glissante_mois": FENETRE,
                "tirages": tirages,
                "graine": GRAINE,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return corps, reel, loi


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--netcdf", default=str(NETCDF_DEFAUT))
    ap.add_argument("--tirages", type=int, default=TIRAGES)
    a = ap.parse_args()
    produire(a.netcdf, tirages=a.tirages)
