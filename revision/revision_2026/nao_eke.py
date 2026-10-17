"""Cross-correlation between the NAO index and basin-mean EKE, with the same
method and thresholds as the WeMO analysis.

Usage: python nao_eke.py --produire
"""

import csv
import io
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
EKE = os.path.join(DOSSIER, "donnees", "eke_mensuelle_trois_champs_1993_2023.csv")
NAO = os.path.join(DOSSIER, "donnees", "nao_cpc_mensuelle_1993_2023.csv")
SORTIE = os.path.join(DOSSIER, "sorties", "nao_eke.csv")
RETARD_MAX = 24
SEUIL_NOMINAL_TAB21 = 0.102555
TESTS = 2 * RETARD_MAX + 1


def desaisonnaliser(valeurs, mois):
    import numpy as np

    v = np.asarray(valeurs, float)
    m = np.asarray(mois)
    out = v.copy()
    for k in range(1, 13):
        out[m == k] -= v[m == k].mean()
    return out


def r1(x):
    import numpy as np

    a = np.asarray(x, float) - np.mean(x)
    return float((a[1:] * a[:-1]).sum() / (a * a).sum())


def n_effectif(x, y):
    a, b = r1(x), r1(y)
    return len(x) * (1 - a * b) / (1 + a * b)


def seuils(n_eff, tests=TESTS):
    import math
    from scipy.stats import norm

    e = (n_eff - 3) ** 0.5
    return (math.tanh(norm.ppf(0.975) / e), math.tanh(norm.ppf(1 - 0.025 / tests) / e))


def correlation_au_retard(indice, energie, k):
    import numpy as np

    x, y = np.asarray(indice, float), np.asarray(energie, float)
    if k > 0:
        x, y = x[:-k], y[k:]
    elif k < 0:
        x, y = x[-k:], y[:k]
    return float(np.corrcoef(x, y)[0, 1])


def lire():
    with io.open(EKE, encoding="utf-8") as f:
        e = list(csv.DictReader(f))
    with io.open(NAO, encoding="utf-8") as f:
        n = {(r["annee"], r["mois"]): float(r["nao_cpc"]) for r in csv.DictReader(f)}
    cles = [(r["annee"], r["mois"]) for r in e]
    if any(c not in n for c in cles):
        raise SystemExit("STOP  EKE months missing from the NAO series")
    eke = [float(r["twosat_025"]) * 1e4 for r in e]
    mois = [int(r["mois"]) for r in e]
    return [n[c] for c in cles], eke, mois


def produire():
    nao, eke, mois = lire()
    anom = desaisonnaliser(eke, mois)
    neff = n_effectif(nao, anom)
    s_nom, s_bon = seuils(neff)
    lignes = []
    for k in range(-RETARD_MAX, RETARD_MAX + 1):
        lignes.append(
            {"retard_mois": k, "r": f"{correlation_au_retard(nao, anom, k):.6f}"}
        )
    r0 = float([l for l in lignes if l["retard_mois"] == 0][0]["r"])
    max12 = max(abs(float(l["r"])) for l in lignes if abs(l["retard_mois"]) <= 12)
    max24 = max(lignes, key=lambda l: abs(float(l["r"])))
    survivent = [l["retard_mois"] for l in lignes if abs(float(l["r"])) > s_bon]
    nominaux = [l["retard_mois"] for l in lignes if abs(float(l["r"])) > s_nom]
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grandeur", "valeur"])
        for g, v in (
            ("n_mois", len(nao)),
            ("r1_nao", f"{r1(nao):.6f}"),
            ("r1_eke", f"{r1(anom):.6f}"),
            ("n_effectif", f"{neff:.6f}"),
            ("r_retard_zero", f"{r0:.6f}"),
            ("max_abs_r_12", f"{max12:.6f}"),
            ("retard_du_max_24", max24["retard_mois"]),
            ("r_max_24", max24["r"]),
            ("seuil_nominal", f"{s_nom:.6f}"),
            ("seuil_bonferroni", f"{s_bon:.6f}"),
            ("retards_nominaux", " ".join(map(str, nominaux)) or "aucun"),
            ("retards_bonferroni", " ".join(map(str, survivent)) or "aucun"),
        ):
            w.writerow([g, v])
        w.writerow([])
        w.writerow(["retard_mois", "r"])
        for l in lignes:
            w.writerow([l["retard_mois"], l["r"]])
    return 0


if __name__ == "__main__":
    if "--produire" in sys.argv:
        sys.exit(produire())
