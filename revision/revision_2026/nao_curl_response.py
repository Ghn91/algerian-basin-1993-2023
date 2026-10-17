"""Monthly NAO index of the NOAA Climate Prediction Center (1993-2023), checked
for completeness, used to repeat the curl-response diagnostic with an index
orthogonal to the WeMO.

Usage: python nao_curl_response.py --telecharger | --verifier
"""

import csv
import hashlib
import json
import os
import sys
import time

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")

URL = (
    "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
    "norm.nao.monthly.b5001.current.ascii"
)
SOURCE = (
    "NOAA Climate Prediction Center, serie NAO mensuelle dediee, "
    "normalisee sur la climatologie 1981-2010"
)
CSV_SORTIE = os.path.join(DONNEES, "nao_cpc_mensuelle_1993_2023.csv")
ARCHIVE = os.path.join(SORTIES, "nao_context.json")

ANNEE_DEBUT, ANNEE_FIN = 1993, 2023

SENTINELLES = (-99.9, -99.90, -999.0, -999.9)
TOLERANCE_SENTINELLE = 1e-6

TENTATIVES = 3
ATTENTE = 5.0


def mois_attendus(annee_debut=ANNEE_DEBUT, annee_fin=ANNEE_FIN):
    return (annee_fin - annee_debut + 1) * 12


def est_une_sentinelle(valeur):
    return any(abs(valeur - s) < TOLERANCE_SENTINELLE for s in SENTINELLES)


def lire_triplets(texte):
    triplets, refus = [], []
    for n, ligne in enumerate(texte.splitlines(), start=1):
        brute = ligne.strip()
        if not brute:
            continue
        jetons = brute.split()
        if len(jetons) != 3:
            refus.append((n, f"{len(jetons)} jetons au lieu de trois"))
            continue
        try:
            an, mo, va = int(jetons[0]), int(jetons[1]), float(jetons[2])
        except ValueError:
            refus.append((n, "un des trois jetons n'est pas un nombre"))
            continue
        if not (1900 <= an <= 2100 and 1 <= mo <= 12):
            refus.append((n, f"annee ou mois hors bornes : {an}-{mo}"))
            continue
        triplets.append((an, mo, va))
    return triplets, refus


def defauts_de_la_serie(triplets, annee_debut=ANNEE_DEBUT, annee_fin=ANNEE_FIN):
    defauts = []

    sentinelles = [(a, m) for a, m, v in triplets if est_une_sentinelle(v)]
    if sentinelles:
        defauts.append(
            f"{len(sentinelles)} valeur(s) sentinelle, la premiere en "
            f"{sentinelles[0][0]}-{sentinelles[0][1]:02d} — REFUSEE, jamais "
            f"absorbee"
        )

    dans = [(a, m, v) for a, m, v in triplets if annee_debut <= a <= annee_fin]
    vus = [(a, m) for a, m, _ in dans]
    if len(set(vus)) != len(vus):
        doubles = sorted({x for x in vus if vus.count(x) > 1})
        defauts.append(
            f"{len(doubles)} mois en double, dont "
            f"{doubles[0][0]}-{doubles[0][1]:02d}"
        )

    attendus = [(a, m) for a in range(annee_debut, annee_fin + 1) for m in range(1, 13)]
    manquants = [x for x in attendus if x not in set(vus)]
    if manquants:
        defauts.append(
            f"{len(manquants)} missing month(s) out of {len(attendus)}, first {manquants[0][0]}-{manquants[0][1]:02d}"
        )

    return defauts


def serie_bornee(triplets, annee_debut=ANNEE_DEBUT, annee_fin=ANNEE_FIN):
    dans = [(a, m, v) for a, m, v in triplets if annee_debut <= a <= annee_fin]
    return sorted(dans, key=lambda t: (t[0], t[1]))


def decimales_utiles(valeur, maximum=6):
    for d in range(maximum + 1):
        if abs(round(valeur, d) - valeur) < 1e-12:
            return d
    return maximum


def changements_de_quantification(texte):
    par_annee = {}
    for ligne in texte.splitlines():
        jetons = ligne.split()
        if len(jetons) != 3:
            continue
        try:
            an, va = int(jetons[0]), float(jetons[2])
        except ValueError:
            continue
        par_annee.setdefault(an, set()).add(decimales_utiles(va))
    annees = sorted(par_annee)
    ruptures = []
    for a, b in zip(annees, annees[1:]):
        if max(par_annee[a]) != max(par_annee[b]):
            ruptures.append((b, max(par_annee[a]), max(par_annee[b])))
    return ruptures


def empreinte(texte):
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


ERREURS_RESEAU = Exception


def telecharger_texte(url=URL, tentatives=TENTATIVES, attente=ATTENTE, ouvrir=None):
    if ouvrir is None:
        from urllib.request import urlopen

        def ouvrir(u):
            with urlopen(u, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace")

    raisons = []
    for essai in range(1, tentatives + 1):
        try:
            texte = ouvrir(url)
        except ERREURS_RESEAU as e:
            raisons.append(f"tentative {essai} : {type(e).__name__} — {e}")
            if essai < tentatives:
                time.sleep(attente)
            continue
        if not texte or not texte.strip():
            raisons.append(f"tentative {essai} : reponse vide")
            if essai < tentatives:
                time.sleep(attente)
            continue
        return texte, raisons
    return None, raisons


def telecharger():
    os.makedirs(DONNEES, exist_ok=True)
    os.makedirs(SORTIES, exist_ok=True)

    texte, raisons = telecharger_texte()
    if texte is None:
        return 1

    triplets, refus = lire_triplets(texte)

    defauts = defauts_de_la_serie(triplets)
    if defauts:
        return 1

    serie = serie_bornee(triplets)
    bornes = (f"{serie[0][0]}-{serie[0][1]:02d}", f"{serie[-1][0]}-{serie[-1][1]:02d}")

    ruptures = changements_de_quantification(texte)

    with open(CSV_SORTIE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["annee", "mois", "nao_cpc"])
        w.writerows([[a, m, f"{v:.4f}"] for a, m, v in serie])

    with open(ARCHIVE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "releve_le": time.strftime("%Y-%m-%d"),
                "url": URL,
                "source": SOURCE,
                "sha256_du_texte_recu": empreinte(texte),
                "lignes_recues": len(texte.splitlines()),
                "triplets_lus": len(triplets),
                "lignes_refusees": len(refus),
                "mois_de_la_periode": len(serie),
                "mois_attendus": mois_attendus(),
                "premier_mois": bornes[0],
                "dernier_mois": bornes[1],
                "derniere_annee_du_fichier": max(a for a, _, _ in triplets),
                "changements_de_quantification": [
                    {
                        "annee": a,
                        "decimales_utiles_avant": av,
                        "decimales_utiles_apres": ap,
                    }
                    for a, av, ap in ruptures
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return 0


def verifier():
    if not os.path.exists(ARCHIVE):
        return 1
    with open(ARCHIVE, encoding="utf-8") as f:
        a = json.load(f)
    if not os.path.exists(CSV_SORTIE):
        return 1
    with open(CSV_SORTIE, newline="", encoding="utf-8") as f:
        n = sum(1 for _ in csv.DictReader(f))
    return 0 if n == a["mois_de_la_periode"] else 1


def main(argv):
    if "--telecharger" in argv:
        return telecharger()
    if "--verifier" in argv:
        return verifier()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
