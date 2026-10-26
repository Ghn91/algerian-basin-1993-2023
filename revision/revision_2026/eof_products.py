"""EOF decomposition of monthly SLA anomalies of the two altimetry products
on a common 0.25 deg grid, as computed and detrended, with North's separability criterion
computed with the effective sample size of each principal component.

Usage: python eof_products.py --produire
"""

import csv
import json
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")
TRANSPORT = os.path.join(DOSSIER, "sorties_these")

sys.path.insert(0, DOSSIER)

from altimetry_products import (
    ALLSAT,
    TWOSAT,
    FACTEUR,
    ANNEE_DEBUT,
    ANNEE_FIN,
    moyenner_blocs,
    masque_commun,
    grilles_coincident,
)

CHAMPS = ("allsat_0125", "allsat_025", "twosat_025")
N_MODES = 4
TRANSPORTEE = os.path.join(TRANSPORT, "these_ch03_tab16_variance_des_modes_eof.csv")


def anomalie_mensuelle(champ, mois):
    import numpy as np

    c = np.asarray(champ, dtype=float)
    m = np.asarray(mois, dtype=int)
    sortie = np.empty_like(c)
    for k in range(1, 13):
        ou = m == k
        if ou.any():
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


def poids_de_surface(lats, masque=None):
    import numpy as np

    la = np.asarray(lats, dtype=float)
    p = np.sqrt(np.cos(np.radians(la)))
    champ = np.repeat(
        p[:, None], (masque.shape[1] if masque is not None else 1), axis=1
    )
    return champ[masque] if masque is not None else p


def eof(serie, poids=None, n_modes=N_MODES):
    import numpy as np

    x = np.asarray(serie, dtype=float)
    x = x - x.mean(axis=0)
    if poids is not None:
        x = x * np.asarray(poids, dtype=float)[None, :]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    variance = s**2
    parts = variance / variance.sum()
    motifs, pcs = [], []
    for k in range(min(n_modes, vt.shape[0])):
        motif = vt[k].copy()
        pc = u[:, k] * s[k]
        j = int(np.abs(motif).argmax())
        if motif[j] < 0:
            motif, pc = -motif, -pc
        motifs.append(motif)
        pcs.append(pc)
    return {
        "parts": parts[:n_modes],
        "valeurs": variance[:n_modes],
        "motifs": np.array(motifs),
        "pcs": np.array(pcs),
        "variance_totale": float(variance.sum()),
    }


def effectif_independant(pc):
    import numpy as np

    a = np.asarray(pc, dtype=float)
    a = a - a.mean()
    n = a.size
    den = (a * a).sum()
    r1 = float((a[1:] * a[:-1]).sum() / den) if den > 0 else 0.0
    r1 = min(max(r1, -0.999), 0.999)
    return float(np.clip(n * (1.0 - r1) / (1.0 + r1), 2.0, float(n))), r1


def erreur_de_north(valeur, n_effectif):
    import numpy as np

    return float(valeur) * float(np.sqrt(2.0 / float(n_effectif)))


def modes_distinguables(valeurs, effectifs):
    sortie = []
    for k in range(len(valeurs) - 1):
        d1 = erreur_de_north(valeurs[k], effectifs[k])
        d2 = erreur_de_north(valeurs[k + 1], effectifs[k + 1])
        ecart = float(valeurs[k] - valeurs[k + 1])
        seuil = max(d1, d2)
        sortie.append(
            {
                "paire": f"EOF{k + 1}/EOF{k + 2}",
                "ecart_des_valeurs_propres": ecart,
                "erreur_de_north": seuil,
                "distinguables": "oui" if ecart > seuil else "non",
                "marge": ecart - seuil,
            }
        )
    return sortie


def correlation_de_motifs(a, b):
    import numpy as np

    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    den = np.sqrt((x * x).sum() * (y * y).sum())
    r = float((x * y).sum() / den) if den > 0 else float("nan")
    return abs(r), r


def serie_sla(chemin, masque_cible, degrader=False):
    import numpy as np
    import xarray as xr

    mois, lignes = [], []
    ds = xr.open_dataset(chemin)
    try:
        for annee in range(ANNEE_DEBUT, ANNEE_FIN + 1):
            tranche = ds.sel(time=str(annee))
            sla = tranche["sla"].values
            temps = tranche["time"].values
            if degrader:
                sla = moyenner_blocs(sla)
            for k in range(1, 13):
                ou = np.array([int(str(t)[5:7]) == k for t in temps])
                if not ou.any():
                    continue
                lignes.append(sla[ou].mean(axis=0)[masque_cible])
                mois.append((annee, k))
    finally:
        ds.close()
    return mois, np.array(lignes)


def produire():
    import numpy as np
    import xarray as xr

    for chemin in (ALLSAT, TWOSAT):
        if not os.path.exists(chemin):
            return 1
    os.makedirs(SORTIES, exist_ok=True)

    a = xr.open_dataset(ALLSAT)
    t = xr.open_dataset(TWOSAT)
    for nom, fin, large in (
        ("latitude", a.latitude.values, t.latitude.values),
        ("longitude", a.longitude.values, t.longitude.values),
    ):
        ok, ecart = grilles_coincident(fin, large)
        if not ok:
            a.close()
            t.close()
            return 1

    valide_fin = np.isfinite(a["sla"].isel(time=0).values)
    valide_large = np.isfinite(t["sla"].isel(time=0).values)
    masque, complets, partiels = masque_commun(valide_fin, valide_large)
    lats_large = t.latitude.values
    a.close()
    t.close()

    masque_fin = np.repeat(np.repeat(masque, FACTEUR, axis=0), FACTEUR, axis=1)
    masque_fin = masque_fin & valide_fin

    series, mois = {}, None
    mois, series["allsat_0125"] = serie_sla(ALLSAT, masque_fin)
    _, series["allsat_025"] = serie_sla(ALLSAT, masque, degrader=True)
    _, series["twosat_025"] = serie_sla(TWOSAT, masque)

    m = np.array([k for _, k in mois])

    manquantes = {n: int(np.isnan(series[n]).sum()) for n in CHAMPS}
    if any(manquantes.values()):
        return 1

    poids_025 = poids_de_surface(lats_large, masque)

    lignes, paires, contexte = [], [], {}
    motifs_025 = {}

    for nom in CHAMPS:
        brut = anomalie_mensuelle(series[nom], m)
        if nom == "allsat_0125":
            import xarray as xr2

            with xr2.open_dataset(ALLSAT) as ds:
                poids = poids_de_surface(ds.latitude.values, masque_fin)
        else:
            poids = poids_025

        for traitement, champ in (
            ("tel quel", brut),
            ("tendance retiree", retirer_tendance(brut)),
        ):
            for ponderation, p in (("ponderee", poids), ("sans poids", None)):
                r = eof(champ, poids=p)
                effs, r1s = zip(*[effectif_independant(pc) for pc in r["pcs"]])
                for k in range(N_MODES):
                    lignes.append(
                        {
                            "champ": nom,
                            "traitement": traitement,
                            "ponderation": ponderation,
                            "mode": f"EOF{k + 1}",
                            "part_de_variance": f"{r['parts'][k]:.6f}",
                            "r1_de_la_pc": f"{r1s[k]:.6f}",
                            "n_effectif": f"{effs[k]:.4f}",
                            "erreur_de_north_relative": f"{erreur_de_north(r['parts'][k], effs[k]):.6f}",
                        }
                    )
                for d in modes_distinguables(list(r["parts"]), list(effs)):
                    paires.append(
                        {
                            "champ": nom,
                            "traitement": traitement,
                            "ponderation": ponderation,
                            "paire": d["paire"],
                            "ecart_des_parts": f"{d['ecart_des_valeurs_propres']:.6f}",
                            "erreur_de_north": f"{d['erreur_de_north']:.6f}",
                            "distinguables": d["distinguables"],
                            "marge": f"{d['marge']:.6f}",
                        }
                    )
                if ponderation == "ponderee" and nom != "allsat_0125":
                    motifs_025[(nom, traitement)] = r["motifs"][0]

    ecrire(os.path.join(SORTIES, "eof_variance.csv"), lignes)
    ecrire(os.path.join(SORTIES, "eof_north.csv"), paires)

    for nom in CHAMPS:
        for tr in ("tel quel", "tendance retiree"):
            v = [
                l
                for l in lignes
                if l["champ"] == nom
                and l["traitement"] == tr
                and l["ponderation"] == "ponderee"
                and l["mode"] == "EOF1"
            ][0]

    for nom in CHAMPS:
        pp = [
            l
            for l in lignes
            if l["champ"] == nom
            and l["traitement"] == "tel quel"
            and l["mode"] == "EOF1"
        ]
        d = {l["ponderation"]: float(l["part_de_variance"]) for l in pp}

    motifs = []
    for tr in ("tel quel", "tendance retiree"):
        cle_a = ("allsat_025", tr)
        cle_t = ("twosat_025", tr)
        if cle_a in motifs_025 and cle_t in motifs_025:
            absolu, brut_r = correlation_de_motifs(motifs_025[cle_a], motifs_025[cle_t])
            motifs.append(
                {
                    "traitement": tr,
                    "correlation_absolue": f"{absolu:.6f}",
                    "correlation_signee": f"{brut_r:.6f}",
                    "lecture": "meme grille, deux constellations",
                }
            )
    ecrire(os.path.join(SORTIES, "eof_patterns.csv"), motifs)

    if os.path.exists(TRANSPORTEE):
        with open(TRANSPORTEE, newline="", encoding="utf-8") as f:
            ref = {l["mode"]: l for l in csv.DictReader(f)}
        for k in range(N_MODES):
            mode = f"EOF{k + 1}"
            ici = [
                l
                for l in lignes
                if l["champ"] == "twosat_025"
                and l["traitement"] == "tel quel"
                and l["ponderation"] == "ponderee"
                and l["mode"] == mode
            ]
            if mode in ref and ici:
                la_bas = float(ref[mode]["part_tel_quel"])
                v = float(ici[0]["part_de_variance"])
                contexte[f"ecart_{mode}_contre_these"] = round(v - la_bas, 6)

    contexte.update(
        {
            "mailles_du_domaine_commun": int(masque.sum()),
            "mailles_grille_fine": int(masque_fin.sum()),
            "mois": len(mois),
            "modes": N_MODES,
            "valeurs_manquantes": manquantes,
            "remplissage_a_zero": "sans objet : le domaine commun est plein",
            "ponderation": "racine de cos(phi) — la variance est ponderee par "
            "cos(phi), non par son carre",
        }
    )
    with open(
        os.path.join(SORTIES, "eof_context.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(contexte, f, indent=2, ensure_ascii=False)
    return 0


def ecrire(chemin, lignes):
    if not lignes:
        return
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)


def main(argv):
    if "--produire" in argv:
        return produire()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
