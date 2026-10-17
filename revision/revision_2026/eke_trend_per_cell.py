"""Sen's slope of the deseasonalised EKE anomaly at every TWO-SAT grid cell;
the basin mean of the cell slopes is checked against the basin-mean trend.

Usage: python eke_trend_per_cell.py --produire
"""

import csv
import io
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(
    DOSSIER,
    "donnees",
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_"
    "4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc",
)
TAB08 = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab08_six_chemins_et_planchers.csv"
)
SORTIE_MAILLES = os.path.join(DOSSIER, "sorties", "eke_trend_per_cell.csv")
SORTIE_BILAN = os.path.join(DOSSIER, "sorties", "eke_trend_per_cell_summary.csv")
VALIDES_MIN = 0.9


def pente_de_sen(y, pas_par_decennie=120.0):
    import numpy as np

    a = np.asarray(y, dtype=float)
    t = np.arange(a.size, dtype=float)
    ok = np.isfinite(a)
    a, t = a[ok], t[ok]
    i, j = np.triu_indices(a.size, k=1)
    return float(np.median((a[j] - a[i]) / (t[j] - t[i]))) * pas_par_decennie


def desaisonnaliser(serie, mois):
    import numpy as np

    s = np.asarray(serie, dtype=float).copy()
    m = np.asarray(mois)
    for k in range(1, 13):
        sel = m == k
        s[sel] -= np.nanmean(s[sel], axis=0)
    return s


def produire():
    import numpy as np
    import xarray as xr

    with xr.open_dataset(FICHIER) as d:
        eke = 0.5 * (d["ugosa"] ** 2 + d["vgosa"] ** 2) * 1e4
        mensuel = eke.resample(time="1MS").mean().load()
    mois = mensuel["time"].dt.month.values
    n = mensuel.sizes["time"]
    champ = mensuel.values
    valides = np.isfinite(champ).mean(axis=0) >= VALIDES_MIN
    vides = ~np.isfinite(champ).any(axis=0)
    partielles = (~vides) & (~valides)
    if partielles.any():
        raise SystemExit(
            f"STOP  {int(partielles.sum())} partially valid cell(s): a validity threshold would decide them"
        )
    champ = np.where(vides[None], 0.0, champ)
    moyenne = champ.mean(axis=0)
    anom = desaisonnaliser(champ, mois)
    champ = np.where(vides[None], np.nan, champ)

    bassin = np.nanmean(np.where(valides, champ, np.nan), axis=(1, 2))
    pente_bassin = pente_de_sen(desaisonnaliser(bassin, mois))
    with io.open(TAB08, encoding="utf-8") as f:
        publiee = [
            float(r["pente_serie_cm2_s2_par_decennie"])
            for r in csv.DictReader(f)
            if r.get("serie") == "anomalie"
        ][0]

    lignes = []
    for i, la in enumerate(mensuel["latitude"].values):
        for j, lo in enumerate(mensuel["longitude"].values):
            if not valides[i, j]:
                continue
            p = pente_de_sen(anom[:, i, j])
            lignes.append(
                {
                    "latitude": f"{la:.3f}",
                    "longitude": f"{lo:.3f}",
                    "eke_moyenne": f"{moyenne[i, j]:.4f}",
                    "pente_cm2_s-2_decennie": f"{p:.4f}",
                    "pente_relative_decennie": f"{p / moyenne[i, j]:.6f}",
                }
            )
    pentes = np.array([float(l["pente_cm2_s-2_decennie"]) for l in lignes])
    duree_decennies = (n - 1) / 120.0
    moyenne_bassin = float(np.nanmean(bassin))
    bilan = {
        "mailles": len(lignes),
        "mois": n,
        "pente_bassin_recalculee": f"{pente_bassin:.4f}",
        "pente_bassin_ch03_tab08": f"{publiee:.4f}",
        "eke_moyenne_bassin": f"{moyenne_bassin:.4f}",
        "changement_relatif_sur_la_periode": f"{pente_bassin * duree_decennies / moyenne_bassin:.6f}",
        "part_mailles_pente_positive": f"{(pentes > 0).mean():.6f}",
        "pente_quartile_25": f"{np.percentile(pentes, 25):.4f}",
        "pente_mediane": f"{np.median(pentes):.4f}",
        "pente_quartile_75": f"{np.percentile(pentes, 75):.4f}",
    }
    os.makedirs(os.path.dirname(SORTIE_MAILLES), exist_ok=True)
    with io.open(SORTIE_MAILLES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    with io.open(SORTIE_BILAN, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grandeur", "valeur"])
        for k, v in bilan.items():
            w.writerow([k, v])
    ecart = abs(pente_bassin - publiee)
    return 0 if ecart < 0.5 else 1


if __name__ == "__main__":
    if "--produire" in sys.argv:
        sys.exit(produire())
