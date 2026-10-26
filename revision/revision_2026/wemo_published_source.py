"""Identification of the published WeMO series used in the original analysis
with the CRU WeMO index (Web_WeMOi-2020.txt, Climatic Research Unit, UEA).

Usage: python wemo_published_source.py --produire
"""

import csv, hashlib, io, os, sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
CRU = os.path.join(DOSSIER, "donnees", "Web_WeMOi-2020.txt")
SERIE = os.path.join(DOSSIER, "sorties_these", "these_ch03_tab18_wemo_publiee_ub.csv")
SORTIE = os.path.join(DOSSIER, "sorties", "wemo_published_source.csv")
TOLERANCE = 0.005


def lire_cru(texte):
    out = {}
    for l in texte.splitlines():
        p = l.split("\t")
        if p and p[0].strip().isdigit() and len(p) >= 13:
            for m in range(12):
                try:
                    v = float(p[m + 1].strip().replace(",", "."))
                except ValueError:
                    continue
                if v > -998:
                    out[f"{int(p[0])}-{m + 1:02d}"] = v
    return out


def comparer(serie, cru, tol=TOLERANCE):
    communs = [k for k in serie if k in cru]
    ecarts = [abs(serie[k] - cru[k]) for k in communs]
    return {
        "mois_serie": len(serie),
        "mois_communs": len(communs),
        "mois_egaux": sum(e <= tol for e in ecarts),
        "ecart_max": max(ecarts) if ecarts else float("nan"),
    }


def produire():
    t = io.open(CRU, encoding="latin-1").read()
    cru = lire_cru(t)
    with io.open(SERIE, encoding="utf-8") as f:
        serie = {r["mois"]: float(r["wemo_ub_publiee"]) for r in csv.DictReader(f)}
    r = comparer(serie, cru)
    r["sha256_fichier_cru"] = hashlib.sha256(open(CRU, "rb").read()).hexdigest()
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["grandeur", "valeur"])
        for k, v in r.items():
            w.writerow([k, f"{v:.6f}" if isinstance(v, float) else v])
    return 0 if r["mois_egaux"] == r["mois_serie"] else 1


if __name__ == "__main__":
    if "--produire" in sys.argv:
        sys.exit(produire())
    print(__doc__)
