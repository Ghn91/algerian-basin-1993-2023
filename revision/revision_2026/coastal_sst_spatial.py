"""Spatial verdict of the coastal SST test, extracted in columns from the
archived output of coastal_sst.py.

Usage: python coastal_sst_spatial.py --produire
"""

import csv, io, os, re, sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(DOSSIER, "sorties", "sst_spatial_test.csv")
SORTIE = os.path.join(DOSSIER, "sorties", "sst_spatial_verdict.csv")
MOTIF = re.compile(r"n=(\d+),r=(-?\d+\.\d+),p=(\d+\.\d+) plancher=(\d+\.\d+)")


def extraire(lignes):
    out = []
    for l in lignes:
        if len(l) > 1 and "VERDICT" in l[1]:
            m = MOTIF.search(",".join(l[2:]))
            if not m:
                raise SystemExit(f"STOP  unreadable verdict: {l}")
            out.append(
                {
                    "largeur_bande_deg": l[0],
                    "n_bandes": m.group(1),
                    "r": m.group(2),
                    "p": m.group(3),
                    "plancher": m.group(4),
                }
            )
    return out


def produire():
    with io.open(SOURCE, encoding="utf-8") as f:
        v = extraire(list(csv.reader(f)))
    with io.open(SORTIE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(v[0]))
        w.writeheader()
        w.writerows(v)
    return 0


if __name__ == "__main__":
    if "--produire" in sys.argv:
        sys.exit(produire())
