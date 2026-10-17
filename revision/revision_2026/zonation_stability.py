"""Stability of the zonation under perturbations of the mean velocity field
bounded by the single/double precision difference of its time mean.

Usage: python zonation_stability.py --produire [--tirages N]
"""

import csv
import io
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties")
sys.path.insert(0, DOSSIER)

from zonation import zonation
from zone_description import (
    DERIVE,
    accord_des_partitions,
    effectifs_reproduits,
    lire_effectifs_publies,
)

SENSIBILITE = os.path.join(SORTIES, "zone_numerical_sensitivity.csv")
SORTIE = os.path.join(SORTIES, "zonation_stability.csv")
TIRAGES_DEFAUT = 20
GRAINE = 32


def amplitudes_mesurees(chemin=SENSIBILITE):
    with io.open(chemin, encoding="utf-8") as f:
        d = {r["grandeur"]: r["valeur"] for r in csv.DictReader(f)}
    return (float(d["ecart_max_u_cm_s"]) / 100.0, float(d["ecart_max_v_cm_s"]) / 100.0)


def perturber(champ, amplitude, rng):
    c = np.asarray(champ, dtype=np.float64)
    bruit = rng.uniform(-amplitude, amplitude, size=c.shape)
    return np.where(np.isfinite(c), c + bruit, np.nan)


def etiquettes_2d(etiq_plat, mer, forme):
    e = np.zeros(forme, dtype=np.int8)
    e[mer] = etiq_plat
    return e


def produire(n_tirages):
    if not os.path.isfile(DERIVE) or not os.path.isfile(SENSIBILITE):
        return 2
    z = np.load(DERIVE, allow_pickle=False)
    u, v = z["u"], z["v"]
    ref = z["etiquettes"].astype(int)
    if effectifs_reproduits(ref, lire_effectifs_publies()):
        return 1
    e0, m0, ch0, n0 = zonation(u, v)
    if effectifs_reproduits(etiquettes_2d(e0, m0, u.shape), lire_effectifs_publies()):
        return 1
    au, av = amplitudes_mesurees()
    rng = np.random.default_rng(GRAINE)
    lignes = []
    for i in range(0, n_tirages + 1):
        if i == 0:
            e, m, ch, n = zonation(u.astype(np.float64), v.astype(np.float64))
        else:
            e, m, ch, n = zonation(perturber(u, au, rng), perturber(v, av, rng))
        e2 = etiquettes_2d(e, m, u.shape)
        ari, n_com = accord_des_partitions(ref, e2)
        ligne = {
            "tirage": i,
            "nature": "float64 sans bruit" if i == 0 else "bruit borne",
            "classes_retenues": int(n),
            "indice_de_rand_ajuste": f"{ari:.6f}",
            "mailles_communes": n_com,
        }
        for k in range(3, 7):
            ligne[f"critere_k{k}"] = f"{ch[k]:.4f}"
        ligne["critere_croit_jusqu_a_la_borne"] = (
            "oui" if all(ch[k] < ch[k + 1] for k in range(3, 6)) else "non"
        )
        for c in range(1, 7):
            ligne[f"effectif_z{c}"] = int((e2 == c).sum())
        lignes.append(ligne)
    os.makedirs(SORTIES, exist_ok=True)
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    temoin = lignes[0]
    bruites = lignes[1:]
    aris = np.array([float(l["indice_de_rand_ajuste"]) for l in bruites])
    ks = [l["classes_retenues"] for l in bruites]
    bilan = [
        ["tirages", n_tirages],
        ["graine", GRAINE],
        ["borne_bruit_u_m_s", f"{au:.6e}"],
        ["borne_bruit_v_m_s", f"{av:.6e}"],
        ["part_six_classes", f"{ks.count(6) / n_tirages:.6f}"],
        [
            "part_critere_croissant",
            f"{sum(l['critere_croit_jusqu_a_la_borne'] == 'oui' for l in bruites) / n_tirages:.6f}",
        ],
        ["temoin_float64_sans_bruit_classes", temoin["classes_retenues"]],
        ["temoin_float64_sans_bruit_rand_ajuste", temoin["indice_de_rand_ajuste"]],
        ["rand_ajuste_min", f"{aris.min():.6f}"],
        ["rand_ajuste_mediane", f"{np.median(aris):.6f}"],
        ["rand_ajuste_max", f"{aris.max():.6f}"],
        ["part_rand_ajuste_egal_a_un", f"{np.mean(aris > 1 - 1e-12):.6f}"],
    ]
    for k in range(3, 7):
        bilan.append([f"part_{k}_classes", f"{ks.count(k) / n_tirages:.6f}"])
    chemin = SORTIE.replace(".csv", "_summary.csv")
    with io.open(chemin, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["grandeur", "valeur"])
        w.writerows(bilan)
    return 0


if __name__ == "__main__":
    if "--produire" in sys.argv:
        n = TIRAGES_DEFAUT
        if "--tirages" in sys.argv:
            n = int(sys.argv[sys.argv.index("--tirages") + 1])
        sys.exit(produire(n))
