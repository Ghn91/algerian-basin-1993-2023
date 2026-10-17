"""Response of the ERA5 wind-stress curl to the WeMO and MO indices, cell by
cell (detrended correlation at lags 0 and +1 month), with a reference
distribution obtained by randomly permuting the months of the index.

Usage: python curl_response.py --produire
"""

import csv
import json
import math
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")
TRANSPORT = os.path.join(DOSSIER, "sorties_these")

TENSION = os.path.join(DONNEES, "era5_tension_mensuelle_1993_2023.nc")
MASQUE = os.path.join(DONNEES, "masque_cotier_medsea.npz")
INDICES = os.path.join(TRANSPORT, "these_ch05_tab23_indices_reconstruits.csv")

ANNEE_DEBUT, ANNEE_FIN = 1993, 2023
RAYON_TERRE = 6_371_000.0

RETARDS = (0, 1)

TIRAGES = 200
QUANTILE = 0.95
GRAINE = 20261005

COURONNE = 1


def metriques(lats, lons):
    import numpy as np

    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    dlat = abs(float(lat[1] - lat[0]))
    dlon = abs(float(lon[1] - lon[0]))
    dy = RAYON_TERRE * math.radians(dlat)
    dx = RAYON_TERRE * np.cos(np.radians(lat)) * math.radians(dlon)
    return dy, dx


def rotationnel(tau_x, tau_y, lats, lons):
    import numpy as np

    lat = np.asarray(lats, dtype=float)
    if lat.size > 1 and lat[1] <= lat[0]:
        raise ValueError("curl: latitude must be increasing")

    dy, dx = metriques(lat, lons)
    tx = np.asarray(tau_x, dtype=float)
    ty = np.asarray(tau_y, dtype=float)
    dty_dx = np.gradient(ty, axis=-1) / dx[:, None]
    dtx_dy = np.gradient(tx, axis=-2) / dy
    return dty_dx - dtx_dy


def anomalie_mensuelle(champ, mois):
    import numpy as np

    c = np.asarray(champ, dtype=float)
    m = np.asarray(mois, dtype=int)
    sortie = np.empty_like(c)
    for k in range(1, 13):
        ou = m == k
        if not ou.any():
            continue
        sortie[ou] = c[ou] - c[ou].mean(axis=0)
    return sortie


def retirer_tendance(x):
    import numpy as np

    a = np.asarray(x, dtype=float)
    n = a.shape[0]
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    plat = a.reshape(n, -1)
    pente = (t[:, None] * plat).sum(0) / (t * t).sum()
    return (plat - t[:, None] * pente).reshape(a.shape)


def autocorrelation_1(x):
    import numpy as np

    a = np.asarray(x, dtype=float)
    plat = a.reshape(a.shape[0], -1)
    plat = plat - plat.mean(0)
    num = (plat[1:] * plat[:-1]).sum(0)
    den = (plat * plat).sum(0)
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return r.reshape(a.shape[1:])


def effectif_bretherton(n, r1_a, r1_b):
    import numpy as np

    p = np.asarray(r1_a) * np.asarray(r1_b)
    neff = n * (1.0 - p) / (1.0 + p)
    return np.clip(neff, 2.0, float(n))


def seuil_de_r(n_eff, alpha=0.05):
    import numpy as np
    from scipy import stats

    ne = np.asarray(n_eff, dtype=float)
    ddl = np.clip(ne - 2.0, 1.0, None)
    t = stats.t.ppf(1.0 - alpha / 2.0, ddl)
    return t / np.sqrt(t * t + ddl)


def correlation_par_maille(indice, champ, retard=0):
    import numpy as np

    i = np.asarray(indice, dtype=float)
    c = np.asarray(champ, dtype=float)
    if retard > 0:
        i, c = i[:-retard], c[retard:]
    elif retard < 0:
        i, c = i[-retard:], c[:retard]
    i = i - i.mean()
    plat = c.reshape(c.shape[0], -1)
    plat = plat - plat.mean(0)
    num = (i[:, None] * plat).sum(0)
    den = np.sqrt((i * i).sum() * (plat * plat).sum(0))
    r = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return r.reshape(c.shape[1:]), i.size


def plus_grand_amas(booleen):
    import numpy as np

    b = np.asarray(booleen, dtype=bool)
    vu = np.zeros_like(b)
    plus_grand = 0
    for depart in zip(*np.nonzero(b)):
        if vu[depart]:
            continue
        pile, taille = [depart], 0
        vu[depart] = True
        while pile:
            y, x = pile.pop()
            taille += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (
                    0 <= yy < b.shape[0]
                    and 0 <= xx < b.shape[1]
                    and b[yy, xx]
                    and not vu[yy, xx]
                ):
                    vu[yy, xx] = True
                    pile.append((yy, xx))
        plus_grand = max(plus_grand, taille)
    return plus_grand


def coherence(r, masque, seuil):
    import numpy as np

    m = np.asarray(masque, dtype=bool)
    s = np.asarray(seuil, dtype=float) * np.ones_like(r)
    atteint = (np.abs(r) >= s) & m
    n_mer = int(m.sum())
    if n_mer == 0:
        return 0.0, 0.0
    amas = max(plus_grand_amas(atteint & (r > 0)), plus_grand_amas(atteint & (r < 0)))
    return float(atteint.sum()) / n_mer, float(amas) / n_mer


def rebattre(indice, rng):
    import numpy as np

    a = np.asarray(indice, dtype=float).copy()
    rng.shuffle(a)
    return a


def plancher(indice, champ, masque, seuil, retard=0, tirages=TIRAGES, graine=GRAINE):
    import numpy as np

    rng = np.random.default_rng(graine)
    parts, amas = [], []
    for _ in range(tirages):
        r, _ = correlation_par_maille(rebattre(indice, rng), champ, retard)
        p, a = coherence(r, masque, seuil)
        parts.append(p)
        amas.append(a)
    return np.array(parts), np.array(amas)


def masque_sur_grille(masque, lats_src, lons_src, lats_cible, lons_cible):
    import numpy as np

    ms = np.asarray(masque, dtype=bool)
    la_s = np.asarray(lats_src, dtype=float)
    lo_s = np.asarray(lons_src, dtype=float)
    la_c = np.asarray(lats_cible, dtype=float)
    lo_c = np.asarray(lons_cible, dtype=float)

    dedans_la = (la_c >= la_s.min()) & (la_c <= la_s.max())
    dedans_lo = (lo_c >= lo_s.min()) & (lo_c <= lo_s.max())
    i = np.abs(la_c[:, None] - la_s[None, :]).argmin(axis=1)
    j = np.abs(lo_c[:, None] - lo_s[None, :]).argmin(axis=1)
    sortie = ms[np.ix_(i, j)].copy()
    sortie[~dedans_la, :] = False
    sortie[:, ~dedans_lo] = False
    return sortie


def eroder(masque, couronnes=1):
    import numpy as np

    m = np.asarray(masque, dtype=bool)
    for _ in range(max(0, couronnes)):
        garde = m.copy()
        garde[1:, :] &= m[:-1, :]
        garde[:-1, :] &= m[1:, :]
        garde[:, 1:] &= m[:, :-1]
        garde[:, :-1] &= m[:, 1:]
        garde[0, :] = garde[-1, :] = False
        garde[:, 0] = garde[:, -1] = False
        m = garde
    return m


def part_de_tendance(x):
    import numpy as np

    a = np.asarray(x, dtype=float)
    reste = retirer_tendance(a)
    n = a.shape[0]
    va = ((a.reshape(n, -1) - a.reshape(n, -1).mean(0)) ** 2).sum(0)
    vr = ((reste.reshape(n, -1) - reste.reshape(n, -1).mean(0)) ** 2).sum(0)
    part = np.divide(va - vr, va, out=np.zeros_like(va), where=va > 0)
    return part.reshape(a.shape[1:])


NAO = os.path.join(DONNEES, "nao_cpc_mensuelle_1993_2023.csv")


def lire_nao(chemin=None, an_max=ANNEE_FIN):
    import numpy as np

    chemin = chemin or NAO
    if not os.path.exists(chemin):
        return None
    with open(chemin, newline="", encoding="utf-8") as f:
        lignes = [
            l for l in csv.DictReader(f) if ANNEE_DEBUT <= int(l["annee"]) <= an_max
        ]
    lignes.sort(key=lambda l: (int(l["annee"]), int(l["mois"])))
    return (
        np.array([float(l["nao_cpc"]) for l in lignes]),
        np.array([int(l["mois"]) for l in lignes]),
    )


def lire_indices(chemin=None, an_max=ANNEE_FIN):
    import numpy as np

    chemin = chemin or INDICES
    with open(chemin, newline="", encoding="utf-8") as f:
        lignes = [
            l for l in csv.DictReader(f) if ANNEE_DEBUT <= int(l["annee"]) <= an_max
        ]
    lignes.sort(key=lambda l: (int(l["annee"]), int(l["mois"])))
    return {
        "wemo": np.array([float(l["wemo_era5"]) for l in lignes]),
        "mo": np.array([float(l["mo_era5"]) for l in lignes]),
        "mois": np.array([int(l["mois"]) for l in lignes]),
        "annee": np.array([int(l["annee"]) for l in lignes]),
    }


def produire():
    import numpy as np
    import xarray as xr

    for chemin in (TENSION, MASQUE, INDICES):
        if not os.path.exists(chemin):
            return 1

    os.makedirs(SORTIES, exist_ok=True)

    ds = xr.open_dataset(TENSION)
    ds = ds.sortby("latitude")
    lats = ds.latitude.values
    lons = ds.longitude.values
    tx = ds["avg_iews"].values
    ty = ds["avg_inss"].values
    temps = ds["valid_time"].values
    ds.close()

    mois = np.array([int(str(t)[5:7]) for t in temps])
    ind = lire_indices()
    if len(ind["wemo"]) != len(mois):
        return 1
    if not np.array_equal(ind["mois"], mois):
        return 1

    rot = rotationnel(tx, ty, lats, lons)

    z = np.load(MASQUE, allow_pickle=True)
    mer = masque_sur_grille(z["mask"], z["lats"], z["lons"], lats, lons)
    mer_large = eroder(mer, COURONNE)

    dy, dx = metriques(lats, lons)

    rot_a = anomalie_mensuelle(rot, mois)
    parts_tend = part_de_tendance(rot_a)

    lignes = []
    verdicts = []

    nao = lire_nao()
    if nao is None:
        noms = ("wemo", "mo")
    elif not np.array_equal(nao[1], mois):
        return 1
    else:
        ind["nao"] = nao[0]
        noms = ("wemo", "mo", "nao")

    for nom_indice in noms:
        brut = ind[nom_indice]
        brut_a = anomalie_mensuelle(brut, mois)

        for traitement in ("tendance retiree", "sans retrait"):
            if traitement == "tendance retiree":
                i_t = retirer_tendance(brut_a)
                c_t = retirer_tendance(rot_a)
            else:
                i_t, c_t = brut_a, rot_a

            r1_i = float(autocorrelation_1(i_t))
            r1_c = autocorrelation_1(c_t)

            for retard in RETARDS:
                r, n = correlation_par_maille(i_t, c_t, retard)
                neff = effectif_bretherton(n, r1_i, r1_c)
                s = seuil_de_r(neff)
                for nom_dom, m in (
                    ("mer", mer),
                    ("mer au-dela d'une couronne", mer_large),
                ):
                    part, amas = coherence(r, m, s)
                    pl_p, pl_a = plancher(i_t, c_t, m, s, retard)
                    q_p = float(np.quantile(pl_p, QUANTILE))
                    q_a = float(np.quantile(pl_a, QUANTILE))
                    verdicts.append(
                        {
                            "indice": nom_indice,
                            "traitement": traitement,
                            "retard_mois": retard,
                            "domaine": nom_dom,
                            "n_mailles": int(m.sum()),
                            "n_mois": n,
                            "r1_indice": f"{r1_i:.6f}",
                            "r1_rotationnel_median": f"{float(np.median(r1_c[m])):.6f}",
                            "n_effectif_median": f"{float(np.median(neff[m])):.6f}",
                            "seuil_de_r_median": f"{float(np.median(s[m])):.6f}",
                            "r_median": f"{float(np.median(r[m])):.6f}",
                            "r_absolu_max": f"{float(np.abs(r[m]).max()):.6f}",
                            "part_au_seuil": f"{part:.6f}",
                            "part_plancher_mediane": f"{float(np.median(pl_p)):.6f}",
                            "part_plancher_quantile": f"{q_p:.6f}",
                            "part_plancher_max": f"{float(pl_p.max()):.6f}",
                            "part_depasse_le_quantile": "oui" if part > q_p else "non",
                            "amas_au_seuil": f"{amas:.6f}",
                            "amas_plancher_mediane": f"{float(np.median(pl_a)):.6f}",
                            "amas_plancher_quantile": f"{q_a:.6f}",
                            "amas_plancher_max": f"{float(pl_a.max()):.6f}",
                            "amas_depasse_le_quantile": "oui" if amas > q_a else "non",
                            "tirages_de_plancher": TIRAGES,
                            "quantile_du_plancher": QUANTILE,
                        }
                    )

                if traitement == "tendance retiree":
                    for iy, ix in [
                        np.unravel_index(np.where(mer, np.abs(r), -1).argmax(), r.shape)
                    ]:
                        lignes.append(
                            {
                                "indice": nom_indice,
                                "retard_mois": retard,
                                "latitude": f"{lats[iy]:.2f}",
                                "longitude": f"{lons[ix]:.2f}",
                                "r": f"{r[iy, ix]:.6f}",
                                "seuil_de_r": f"{s[iy, ix]:.6f}",
                                "n_effectif": f"{neff[iy, ix]:.6f}",
                                "role": "maille de |r| maximal en mer",
                            }
                        )

    prepares = {n: retirer_tendance(anomalie_mensuelle(ind[n], mois)) for n in noms}
    croisees = {}
    for i, a in enumerate(noms):
        for b in noms[i + 1 :]:
            r_ab = float(np.corrcoef(prepares[a], prepares[b])[0, 1])
            croisees[f"{a}_{b}"] = round(r_ab, 6)
    r_wo = croisees.get("wemo_mo")

    ecrire_csv(os.path.join(SORTIES, "curl_verdict.csv"), verdicts)
    ecrire_csv(os.path.join(SORTIES, "curl_cells.csv"), lignes)
    with open(
        os.path.join(SORTIES, "curl_context.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(
            {
                "indices_testes": list(noms),
                "correlations_croisees_tendance_retiree": croisees,
                "correlation_wemo_mo_tendance_retiree": r_wo,
                "mailles_de_mer": int(mer.sum()),
                "mailles_de_mer_au_dela_d_une_couronne": int(mer_large.sum()),
                "dx_km_min": round(float(dx.min() / 1000), 4),
                "dx_km_max": round(float(dx.max() / 1000), 4),
                "dy_km": round(float(dy / 1000), 4),
                "part_de_tendance_du_rotationnel_mediane": round(
                    float(np.median(parts_tend[mer])), 6
                ),
                "tirages_de_plancher": TIRAGES,
                "graine": GRAINE,
                "retards_testes": list(RETARDS),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return 0


def ecrire_csv(chemin, lignes):
    if not lignes:
        return
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)


def main(argv):
    if "--produire" in argv:
        return produire()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
