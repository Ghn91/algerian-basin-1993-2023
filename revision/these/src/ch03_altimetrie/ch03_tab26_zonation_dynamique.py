"""Zonation of the mean velocity field: Calinski-Harabasz criterion over a
wide range of class numbers, with and without standardisation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))

import pandas as pd
from provenance import sauver_table

NETCDF = RACINE / "data" / "raw" / "medsea_phy-cur_1993-2024.nc"
DERIVE = RACINE / "data" / "interim" / "champ_moyen_medsea_1993_2023.npz"
ARCHIVE = RACINE / "docs" / "ch03_zonation.json"

AN_MAX = 2023
BALAYAGE = range(2, 21)
BORNE_ARTICLE = (3, 6)

PUBLIE = {
    "n_classes": 6,
    "ch_au_maximum": 11056,
    "divergence_zone1_s_moins_1": 1.18e-8,
    "vitesses_cm_s": {1: 15.3, 2: 21.5, 3: 6.8, 4: 4.1, 5: 14.6},
    "u_zone2_cm_s": 20.5,
    "v_zone1_cm_s": 7.9,
    "v_zone5_cm_s": -12.6,
}


def divergence(u, v, lats, lons):
    dlat_m = float(np.abs(np.diff(lats).mean()) * 111000)
    dlon_m = float(
        np.abs(np.diff(lons).mean()) * 111000 * np.cos(np.radians(np.mean(lats)))
    )
    u_fg = np.where(np.isnan(u), 0.0, u)
    v_fg = np.where(np.isnan(v), 0.0, v)
    d = np.gradient(u_fg, dlon_m, axis=1) + np.gradient(v_fg, dlat_m, axis=0)
    return np.where(np.isnan(u) | np.isnan(v), np.nan, d)


def classifier(u_plat, v_plat, k):
    X = StandardScaler().fit_transform(np.column_stack([u_plat, v_plat]))
    Z = linkage(X, method="ward")
    return X, Z, fcluster(Z, t=k, criterion="maxclust")


def balayer(u_plat, v_plat, valeurs=BALAYAGE, standardiser=True):
    brut = np.column_stack([u_plat, v_plat])
    X = StandardScaler().fit_transform(brut) if standardiser else brut
    Z = linkage(X, method="ward")
    sortie = []
    for k in valeurs:
        lbl = fcluster(Z, t=k, criterion="maxclust")
        sortie.append(
            {
                "k": int(k),
                "calinski_harabasz": float(calinski_harabasz_score(X, lbl)),
                "classes_obtenues": int(len(np.unique(lbl))),
            }
        )
    return sortie


def lire_la_courbe(courbe, k_cible):
    vals = [L["calinski_harabasz"] for L in courbe]
    ks = [L["k"] for L in courbe]
    if k_cible not in ks:
        return None
    i = ks.index(k_cible)
    ici = vals[i]
    gauche = vals[i - 1] if i > 0 else None
    droite = vals[i + 1] if i < len(vals) - 1 else None
    local = (gauche is None or ici > gauche) and (droite is None or ici > droite)
    voisines = [x for x in (gauche, droite) if x is not None]
    marge = (ici - max(voisines)) / ici if (local and voisines) else float("nan")
    return {
        "k": k_cible,
        "calinski_harabasz": ici,
        "maximum_local": "oui" if local else "non",
        "marge_relative_sur_les_voisines": marge,
        "rang_dans_le_balayage": int(sum(1 for x in vals if x > ici) + 1),
        "valeurs_superieures": int(sum(1 for x in vals if x > ici)),
        "valeurs_balayees": len(vals),
    }


def maximum_interieur(courbe):
    vals = [L["calinski_harabasz"] for L in courbe]
    i = int(np.argmax(vals))
    return courbe[i]["k"], 0 < i < len(vals) - 1


def renommer_par_vitesse(etiquettes, u_plat, v_plat):
    spd = np.sqrt(np.asarray(u_plat) ** 2 + np.asarray(v_plat) ** 2)
    ordre = sorted(
        np.unique(etiquettes), key=lambda c: -float(np.nanmean(spd[etiquettes == c]))
    )
    table = {ancien: rang + 1 for rang, ancien in enumerate(ordre)}
    return np.array([table[c] for c in etiquettes]), table


def statistiques(etiquettes, u_plat, v_plat, div_plat):
    lignes = []
    spd = np.sqrt(np.asarray(u_plat) ** 2 + np.asarray(v_plat) ** 2)
    for c in sorted(np.unique(etiquettes)):
        m = etiquettes == c
        lignes.append(
            {
                "unite": f"Unite {int(c)}",
                "effectif": int(m.sum()),
                "part_du_domaine": float(m.sum() / len(etiquettes)),
                "u_moyen_cm_s": float(np.nanmean(u_plat[m]) * 100),
                "v_moyen_cm_s": float(np.nanmean(v_plat[m]) * 100),
                "vitesse_moyenne_cm_s": float(np.nanmean(spd[m]) * 100),
                "divergence_1e-6_s-1": float(np.nanmean(div_plat[m]) * 1e6),
                "tendance": (
                    "divergence" if np.nanmean(div_plat[m]) > 0 else "convergence"
                ),
            }
        )
    return lignes


def dispositif(lons, lats, s_u, s_v):
    return [
        {
            "grandeur": "longitude minimale du domaine",
            "unite": "degres est",
            "emprise_de_la_these": float(lons.min()),
            "emprise_de_l_article": -5.00,
        },
        {
            "grandeur": "longitude maximale du domaine",
            "unite": "degres est",
            "emprise_de_la_these": float(lons.max()),
            "emprise_de_l_article": 10.00,
        },
        {
            "grandeur": "latitude minimale du domaine",
            "unite": "degres nord",
            "emprise_de_la_these": float(lats.min()),
            "emprise_de_l_article": 35.02,
        },
        {
            "grandeur": "latitude maximale du domaine",
            "unite": "degres nord",
            "emprise_de_la_these": float(lats.max()),
            "emprise_de_l_article": 40.98,
        },
        {
            "grandeur": "ecart-type de la composante zonale moyenne",
            "unite": "m/s",
            "emprise_de_la_these": s_u,
            "emprise_de_l_article": float("nan"),
        },
        {
            "grandeur": "ecart-type de la composante meridienne moyenne",
            "unite": "m/s",
            "emprise_de_la_these": s_v,
            "emprise_de_l_article": float("nan"),
        },
        {
            "grandeur": "rapport des deux ecarts-types",
            "unite": "sans",
            "emprise_de_la_these": s_u / s_v,
            "emprise_de_l_article": float("nan"),
        },
    ]


def regard_sur_le_manuscrit(courbe, stats):
    au_nombre_retenu = next(
        L["calinski_harabasz"] for L in courbe if L["k"] == PUBLIE["n_classes"]
    )
    divergentes = [L for L in stats if L["divergence_1e-6_s-1"] > 0]
    plus_rapide = max(stats, key=lambda L: L["vitesse_moyenne_cm_s"])
    return [
        {
            "grandeur": "critere au nombre retenu",
            "reference": "-",
            "publie_au_manuscrit": float(PUBLIE["ch_au_maximum"]),
            "mesure_par_la_these": au_nombre_retenu,
            "remarque": "emprises differentes : les valeurs ne se comparent pas "
            "terme a terme",
        },
        {
            "grandeur": "divergence de l unite la plus divergente",
            "reference": "zone 1 du manuscrit",
            "publie_au_manuscrit": PUBLIE["divergence_zone1_s_moins_1"] * 1e6,
            "mesure_par_la_these": max(L["divergence_1e-6_s-1"] for L in stats),
            "remarque": "en 1e-6 s-1",
        },
        {
            "grandeur": "vitesse de l unite la plus rapide",
            "reference": "zone 2 du manuscrit",
            "publie_au_manuscrit": float(PUBLIE["vitesses_cm_s"][2]),
            "mesure_par_la_these": plus_rapide["vitesse_moyenne_cm_s"],
            "remarque": "cm/s",
        },
        {
            "grandeur": "composante zonale de l unite la plus rapide",
            "reference": "zone 2 du manuscrit",
            "publie_au_manuscrit": float(PUBLIE["u_zone2_cm_s"]),
            "mesure_par_la_these": plus_rapide["u_moyen_cm_s"],
            "remarque": "cm/s",
        },
        {
            "grandeur": "unites a divergence moyenne positive",
            "reference": "-",
            "publie_au_manuscrit": 1.0,
            "mesure_par_la_these": float(len(divergentes)),
            "remarque": "le manuscrit en declare une seule ; ecart de fond, mais "
            "sur une autre emprise, donc pas une contradiction",
        },
    ]


def champ_moyen(chemin=NETCDF, an_max=AN_MAX):
    import xarray as xr

    ds = xr.open_dataset(chemin)
    try:
        ds = ds.sel(time=slice(None, f"{an_max}-12-31"))
        u = ds["uo"].mean("time").squeeze().values
        v = ds["vo"].mean("time").squeeze().values
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        n_pas = int(ds.sizes["time"])
        bornes = (str(ds["time"].values[0])[:10], str(ds["time"].values[-1])[:10])
    finally:
        ds.close()
    return u, v, lats, lons, n_pas, bornes


def ecrire_derive(chemin, u, v, lats, lons, bornes):
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chemin, u=u, v=v, latitude=lats, longitude=lons, bornes=np.array(bornes)
    )


def lire_derive(chemin=DERIVE):
    p = Path(chemin)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return (
        d["u"],
        d["v"],
        d["latitude"],
        d["longitude"],
        tuple(str(x) for x in d["bornes"]),
    )


def produire():
    archive = lire_derive()
    if NETCDF.exists():
        u, v, lats, lons, n_pas, bornes = champ_moyen()
        if archive is not None:
            ecart = float(np.nanmax(np.abs(u - archive[0])))
            if ecart > 1e-9:
                raise SystemExit(
                    f"  the archive and the computation differ by {ecart:.3e}."
                )
        ecrire_derive(DERIVE, u, v, lats, lons, bornes)
    elif archive is not None:
        u, v, lats, lons, bornes = archive
    else:
        raise SystemExit(f"ni {NETCDF} ni {DERIVE}.")

    div = divergence(u, v, lats, lons)
    valide = ~(np.isnan(u) | np.isnan(v))
    u_plat, v_plat, div_plat = u[valide], v[valide], div[valide]

    s_u, s_v = float(np.std(u_plat)), float(np.std(v_plat))

    courbe = balayer(u_plat, v_plat)
    sans = balayer(u_plat, v_plat, standardiser=False)
    k_max, interieur = maximum_interieur(courbe)
    k_sans, interieur_sans = maximum_interieur(sans)
    for L, S in zip(courbe, sans):
        L["calinski_harabasz_sans_standardisation"] = S["calinski_harabasz"]

    fenetre = [L for L in courbe if BORNE_ARTICLE[0] <= L["k"] <= BORNE_ARTICLE[1]]
    k_article = max(fenetre, key=lambda L: L["calinski_harabasz"])["k"]

    for L in courbe:
        m = lire_la_courbe(courbe, L["k"])
        L["maximum_local"] = m["maximum_local"]
        L["marge_relative_sur_les_voisines"] = m["marge_relative_sur_les_voisines"]
        L["rang_dans_le_balayage"] = m["rang_dans_le_balayage"]

    _, _, brut = classifier(u_plat, v_plat, PUBLIE["n_classes"])
    etiquettes, correspondance = renommer_par_vitesse(brut, u_plat, v_plat)
    stats = statistiques(etiquettes, u_plat, v_plat, div_plat)

    SCRIPT = "src/ch03_altimetrie/ch03_tab26_zonation_dynamique.py"
    DOM = "bassin-algerien-1993-2023"
    SRC = NETCDF.name if NETCDF.exists() else DERIVE.name

    def texte(rows):
        return pd.DataFrame(
            [
                {c: (f"{v:.6f}" if isinstance(v, float) else v) for c, v in L.items()}
                for L in rows
            ]
        )

    sauver_table(
        texte(courbe),
        "ch03_tab26",
        "balayage du critere de calinski harabasz de 2 a 20 classes",
        SCRIPT,
        SRC,
        "3.8",
        domaine=DOM,
    )
    sauver_table(
        texte(stats),
        "ch03_tab27",
        "statistiques des unites de la zonation dynamique",
        SCRIPT,
        SRC,
        "3.8",
        domaine=DOM,
    )

    sauver_table(
        texte(dispositif(lons, lats, s_u, s_v)),
        "ch03_tab29",
        "emprises et echelles du dispositif de zonation",
        SCRIPT,
        SRC,
        "3.8",
        domaine=DOM,
    )
    sauver_table(
        texte(regard_sur_le_manuscrit(courbe, stats)),
        "ch03_tab28",
        "valeurs de la zonation publiees au manuscrit et mesurees par " "la these",
        SCRIPT,
        SRC,
        "3.8",
        domaine=DOM,
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "produit": "MEDSEA PHY courants 1/24 deg, uo et vo a 1,02 m — le "
                "produit des chapitres 4 et 5, non l'altimetrie",
                "descripteurs": "moyenne temporelle de u et de v, standardisees — deux, "
                "et deux seulement ; la divergence decrit les classes "
                "apres coup et n'entre pas dans la classification",
                "emprise_these": {
                    "lon": [float(lons.min()), float(lons.max())],
                    "lat": [float(lats.min()), float(lats.max())],
                },
                "emprise_article": {"lon": [-5.00, 10.00], "lat": [35.02, 40.98]},
                "replication": "non : les deux emprises ne coincident pas",
                "periode": list(bornes),
                "mailles_valides": int(valide.sum()),
                "balayage": [BALAYAGE.start, BALAYAGE.stop - 1],
                "maximum": {"k": k_max, "interieur": bool(interieur)},
                "maximum_sans_standardisation": {
                    "k": k_sans,
                    "interieur": bool(interieur_sans),
                },
                "ecarts_types_du_champ_moyen": {
                    "u": s_u,
                    "v": s_v,
                    "rapport": s_u / s_v,
                },
                "maximum_sur_la_fenetre_de_l_article": k_article,
                "publie_au_manuscrit": PUBLIE,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return courbe, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    produire()
