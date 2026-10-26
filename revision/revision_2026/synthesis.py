"""Helper functions of the synthesis figure (Figure 9), which is drawn by
figures.py.
"""

import csv
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties")
THESE = os.path.join(DOSSIER, "sorties_these")
DONNEES = os.path.join(DOSSIER, "donnees")

SERIE = os.path.join(DONNEES, "eke_mensuelle_trois_champs_1993_2023.csv")
TENDANCES = os.path.join(SORTIES, "products_trends.csv")
VARIANCE = os.path.join(SORTIES, "figure_05_variance_shares.csv")
NORTH = os.path.join(SORTIES, "eof_north.csv")
BALAYAGE = os.path.join(THESE, "these_ch03_tab20_balayage_des_retards.csv")
VERDICT = os.path.join(THESE, "these_ch03_tab21_verdict_et_effectif.csv")
BANDES = os.path.join(SORTIES, "ekman_africa_index_by_band.csv")


SERIE_C = "reconstruction d'origine (signe inverse)"
SIGNE_C = -1.0
ENERGIES_C = ("totale_quotidienne", "turbulente_quotidienne")


def lire(chemin):
    with open(chemin, newline="", encoding="utf-8") as f:
        return [l for l in csv.DictReader(f)]


def annuelle(lignes, colonne):
    par_an = {}
    for l in lignes:
        try:
            an, v = int(l["annee"]), float(l[colonne])
        except (KeyError, ValueError):
            continue
        par_an.setdefault(an, []).append(v)
    ans = sorted(par_an)
    return ans, [sum(par_an[a]) / len(par_an[a]) for a in ans]


def retards(lignes, energie, serie):
    pts = [
        (int(l["retard_mois"]), float(l["r"]))
        for l in lignes
        if l["energie"] == energie and l["serie"] == serie
    ]
    return sorted(pts)


def separabilite(lignes, champ, traitement="tel quel", ponderation="ponderee"):
    return [
        (
            l["paire"],
            float(l["ecart_des_parts"]),
            float(l["erreur_de_north"]),
            l["distinguables"] == "oui",
        )
        for l in lignes
        if l["champ"] == champ
        and l["traitement"] == traitement
        and l["ponderation"] == ponderation
    ]


def contigues(valeurs, positif=True):
    meilleur, courant, debut_c = (0, 0), 0, 0
    for i, v in enumerate(list(valeurs) + [None]):
        ok = v is not None and ((v > 0) if positif else (v < 0))
        if ok:
            if courant == 0:
                debut_c = i
            courant += 1
            if courant > meilleur[1]:
                meilleur = (debut_c, courant)
        else:
            courant = 0
    return meilleur


def produire(apercu=None):
    sys.path.insert(0, DOSSIER)
    import figures as fr

    return fr.figure_9(apercu)


def main(argv):
    if "--produire" in argv:
        return produire()
    if "--apercu" in argv:
        i = argv.index("--apercu")
        return produire(apercu=argv[i + 1])
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
