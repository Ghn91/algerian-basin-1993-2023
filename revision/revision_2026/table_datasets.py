"""LaTeX tables of dataset provenance and extraction, generated from the
outputs of dataset_records.py.

Usage: python table_datasets.py --construire
"""

import csv
import io
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties")
CIBLE = os.path.join(DOSSIER, "manuscript", "tableau_des_jeux.tex")

FICHES = os.path.join(SORTIES, "datasets_records.csv")
EXTRACTION = os.path.join(SORTIES, "datasets_extraction.csv")
RECUS = os.path.join(SORTIES, "datasets_files_received.csv")

ETIQUETTES = {
    "SEALEVEL_GLO_PHY_L4_MY_008_047": {
        "court": "ALL-SAT altimetry",
        "statut": "employe",
        "role": "SLA, geostrophic velocity anomalies; EOF and EKE fields",
    },
    "TWO-SAT": {
        "court": "TWO-SAT altimetry",
        "statut": "employe",
        "role": "stable-constellation product; trends and inter-epoch " "comparison",
    },
    "SEALEVEL_EUR_PHY_MDT_L4_STATIC_008_070": {
        "court": "Mean dynamic topography",
        "statut": "employe",
        "role": "mean geostrophic circulation",
    },
    "MEDSEA_MULTIYEAR_PHY_006_004": {
        "court": "MEDSEA physical reanalysis",
        "statut": "employe",
        "role": "surface currents for the dynamic zonation; temperature, "
        "salinity, sea level, mixed-layer depth",
    },
    "SST_MED_SST_L4_REP_OBSERVATIONS_010_021": {
        "court": "Satellite sea surface temperature",
        "statut": "employe",
        "role": "coastal SST response",
    },
    "MEDSEA_MULTIYEAR_BGC_006_008": {
        "court": "Biogeochemical reanalysis",
        "statut": "employe",
        "role": "biogeochemical contrast between regimes",
    },
    "reanalysis-era5-single-levels-monthly-means": {
        "court": "ERA5 atmospheric reanalysis",
        "statut": "employe",
        "role": "wind stress and curl, mean sea-level pressure, "
        "reconstructed WeMO index",
    },
    "GLOBAL_MULTIYEAR_PHY_001_030": {
        "court": "GLORYS12 global reanalysis",
        "statut": "retire",
        "role": "retrieved but not used; the comparison described in the "
        "submitted version was not carried out",
    },
    "INSITU_MED_PHYBGCWAV_DISCRETE_MYNRT_013_035": {
        "court": "In situ observations",
        "statut": "retire",
        "role": "retrieved but not used; the validation described in the "
        "submitted version was not carried out",
        "grille": "not gridded (individual profiles)",
    },
}

OBLIGATOIRES = {
    "provenance": ("court", "identifiant", "fournisseur"),
    "usage": ("court", "grille", "cadence", "periode", "variables", "niveau"),
}


def echapper(s):
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def version_du_jeu(jeux, dataset_id):
    import re

    if not dataset_id:
        return "", "aucun identifiant de jeu dans l'appel"
    cands = [j.strip() for j in jeux if j.strip().startswith(dataset_id)]
    if not cands:
        return "", "aucun jeu du catalogue ne commence par cet identifiant"
    vus = set()
    for c in cands:
        m = re.match(r"_?(\d{6})", c[len(dataset_id) :])
        vus.add(m.group(1) if m else "")
    if len(vus) > 1:
        return "", (
            "les " + str(len(cands)) + " jeux correspondants portent "
            "des millesimes differents : " + ", ".join(sorted(vus))
        )
    v = vus.pop()
    return (v, "") if v else ("", "le jeu ne porte pas de millesime")


def niveau_lisible(dmin, dmax):
    if dmin in (None, "") and dmax in (None, ""):
        return "sea surface"
    try:
        a, b = float(dmin), float(dmax)
    except (TypeError, ValueError):
        return ""
    if abs(a - b) < 1e-9:
        return (
            f"first model level, nominally \\num{{{a:.2f}}}~m"
            if a > 0.01
            else "sea surface"
        )
    return (
        f"\\num{{{a:.0f}}} to \\num{{{b:.0f}}}~m, that is at most the "
        f"uppermost \\num{{{max(abs(a), abs(b)):.0f}}}~m"
    )


def grille_lisible(fiche, recu, declaree=""):
    p = (recu or {}).get("pas_lon_deg", "").strip()
    q = (recu or {}).get("pas_lat_deg", "").strip()
    if not p:
        r = (fiche or {}).get("resolution_spatiale", "").strip()
        if r:
            import re as _re

            m = _re.fullmatch(r"([\d.]+)\s*degrees?", r)
            return f"\\num{{{m.group(1)}}}\\textdegree{{}}" if m else r
        return declaree
    a, b = float(p), float(q or p)

    def _d(v):
        return f"{v:.4f}".rstrip("0").rstrip(".")

    return (
        f"\\num{{{_d(a)}}}\\textdegree{{}}"
        if _d(a) == _d(b)
        else f"\\num{{{_d(a)}}} $\\times$ \\num{{{_d(b)}}}" f"\\textdegree{{}}"
    )


def cadence_lisible(fiche, recu):
    j = (recu or {}).get("cadence_jours", "").strip()
    if j:
        v = float(j)
        if abs(v - 1.0) < 0.01:
            return "daily"
        if 27.9 <= v <= 31.1:
            return "monthly"
        return f"\\num{{{v:.2f}}}~d"
    return (fiche or {}).get("resolutions_temporelles", "").strip()


def agreger(valeurs):
    out = []
    for v in valeurs:
        for x in str(v or "").split(";"):
            x = x.strip()
            if x and x not in out:
                out.append(x)
    return "; ".join(out)


def periode_lisible(extraits, recus, fiche):
    bornes = []
    for r in recus or []:
        p = (r.get("periode_portee") or "").strip()
        if " -> " in p:
            bornes.append(tuple(p.split(" -> ")))
    if not bornes:
        for x in extraits or []:
            d, f = x.get("debut_extrait", ""), x.get("fin_extrait", "")
            if d and f:
                bornes.append((d, f))
    if bornes:
        return f"{min(b[0] for b in bornes)}--{max(b[1] for b in bornes)}"
    a = (fiche or {}).get("periode_debut", "").strip()
    b = (fiche or {}).get("periode_fin", "").strip()
    if a and b:
        return f"static field, {a}" if a == b else f"{a}--{b}"
    return ""


def variables_lisibles(extrait):
    v = [x.strip() for x in (extrait or "").split(";") if x.strip()]
    return ", ".join("\\texttt{" + echapper(x) + "}" for x in v)


def lignes_du_tableau(fiches, extraits, recus, etiquettes=None):
    et = ETIQUETTES if etiquettes is None else etiquettes
    prov, usage, defauts = [], [], []

    par_produit = {f.get("produit", ""): f for f in fiches}
    inconnus = [p for p in par_produit if p and p not in et]
    for p in inconnus:
        defauts.append(f"produit RELEVE sans etiquette declaree : {p}")
    for p in et:
        if p not in par_produit:
            defauts.append(f"etiquette ORPHELINE, aucun produit releve : {p}")

    rang = {"employe": 0, "retire": 1}
    for p, meta in sorted(
        et.items(), key=lambda kv: (rang.get(kv[1]["statut"], 9), kv[1]["court"])
    ):
        f = par_produit.get(p)
        if f is None:
            continue
        jeux = [j for j in (f.get("jeux") or "").split(";") if j.strip()]

        def du_produit(rangs):
            return [
                y
                for y in rangs
                if y.get("jeu") and any(j.strip().startswith(y["jeu"]) for j in jeux)
            ]

        xs, rs = du_produit(extraits), du_produit(recus)
        dsid = (xs + rs)[0]["jeu"] if (xs or rs) else ""
        ver, pourquoi = version_du_jeu(jeux, dsid)
        r = rs[0] if rs else {}
        if pourquoi and dsid and "ne porte pas de millesime" in pourquoi:
            ver = "n/a"
        elif pourquoi and dsid:
            defauts.append(f"{meta['court']} : version non rendue — " f"{pourquoi}")

        prov.append(
            {
                "court": meta["court"],
                "statut": meta["statut"],
                "identifiant": (
                    p
                    if p in par_produit and f.get("titre", "")[:1] != "["
                    else (jeux[0].strip() if jeux else p)
                ),
                "version": ver,
                "fournisseur": f.get("producteur", "").strip(),
                "doi": f.get("doi", "").strip() or r.get("doi", "").strip(),
            }
        )
        usage.append(
            {
                "court": meta["court"],
                "statut": meta["statut"],
                "grille": grille_lisible(f, r, meta.get("grille", "")),
                "cadence": cadence_lisible(f, r),
                "periode": periode_lisible(xs, rs, f),
                "variables": variables_lisibles(
                    agreger([y.get("variables_portees") for y in rs])
                    or agreger([y.get("variables_extraites") for y in xs])
                ),
                "niveau": "; ".join(
                    dict.fromkeys(
                        [
                            niveau_lisible(
                                y.get("profondeur_min_m"), y.get("profondeur_max_m")
                            )
                            for y in (rs or xs)
                        ]
                        or ["sea surface"]
                    )
                ),
                "role": meta["role"],
            }
        )

    for nom, cols in OBLIGATOIRES.items():
        for l in (prov if nom == "provenance" else usage):
            for c in cols:
                if not str(l.get(c, "")).strip():
                    defauts.append(
                        f"{nom} / {l.get('court', '?')} : la colonne « {c} » est VIDE"
                    )
    return prov, usage, defauts


def ident(s):
    s = str(s)
    if not s or s == "---":
        return "---"
    for c in "{}%#\\":
        if c in s:
            raise ValueError(
                f"caractere non pris en charge dans un identifiant : {s!r}"
            )
    return r"\nolinkurl{" + s + "}"


def ligne(cellules):
    return (
        "    "
        + " & ".join(r"\raggedright\hspace{0pt}" + c for c in cellules)
        + r" \tabularnewline"
    )


def tableau_provenance(lignes):
    out = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Provenance of the datasets used in this study. "
        r"Product identifiers, versions, providers and digital object "
        r"identifiers are those published by the data providers' own "
        r"catalogues, retrieved from the Copernicus Marine Service and "
        r"Climate Data Store catalogue interfaces rather than "
        r"transcribed. The last two entries were retrieved but are not "
        r"used in this study (Section~\ref{sec:hca}).}",
        r"  \label{tab:datasets-provenance}",
        r"  \begin{tabular}{@{}p{2.7cm}p{4.9cm}p{1.2cm}p{2.3cm}p{3.6cm}@{}}",
        r"    \toprule",
        ligne(
            [
                "Dataset",
                "Product or collection identifier",
                "Version",
                "Provider",
                "DOI",
            ]
        ),
        r"    \midrule",
    ]
    statut = None
    for l in lignes:
        if statut is not None and l["statut"] != statut:
            out.append(r"    \midrule")
        statut = l["statut"]
        out.append(
            ligne(
                [
                    echapper(l["court"]),
                    ident(l["identifiant"]),
                    echapper(l["version"]) or "---",
                    echapper(l["fournisseur"]),
                    ident(l["doi"]) if l["doi"] else "---",
                ]
            )
        )
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def tableau_usage(lignes):
    out = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Spatial and temporal resolution, period, variables "
        r"and vertical level of the data actually extracted for each "
        r"dataset. Grid spacings are those of the coordinate axes of the "
        r"retrieved files, each computed as the extent of the axis divided "
        r"by its number of intervals, and agree with the "
        r"providers' dataset-level declarations; where the two axes of a "
        r"product are not equally spaced, both are given, which is why "
        r"some entries differ slightly from the single nominal figure "
        r"quoted in the product catalogues. Periods are those of the "
        r"data retrieved; variables are those retrieved, under the names "
        r"they carry in the downloaded files, and of the MEDSEA physical "
        r"reanalysis only the velocity components enter the analyses of "
        r"this paper. Model levels are given by the nominal depth of the "
        r"first level.}",
        r"  \label{tab:datasets-usage}",
        r"  \begin{tabular}{@{}p{2.5cm}p{1.7cm}p{1.7cm}p{1.8cm}p{3.7cm}p{2.9cm}@{}}",
        r"    \toprule",
        ligne(
            ["Dataset", "Grid", "Cadence", "Period used", "Variables", "Vertical level"]
        ),
        r"    \midrule",
    ]
    statut = None
    for l in lignes:
        if statut is not None and l["statut"] != statut:
            out.append(r"    \midrule")
        statut = l["statut"]
        out.append(
            ligne(
                [
                    echapper(l["court"]),
                    l["grille"],
                    echapper(l["cadence"]),
                    l["periode"],
                    l["variables"],
                    l["niveau"],
                ]
            )
        )
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out)


def lire(chemin):
    if not os.path.isfile(chemin):
        return None
    return list(csv.DictReader(io.open(chemin, encoding="utf-8")))


def construire(ecrire=True):
    fiches, extraits, recus = lire(FICHES), lire(EXTRACTION), lire(RECUS)
    manquantes = [
        os.path.basename(c)
        for c, v in ((FICHES, fiches), (EXTRACTION, extraits))
        if v is None
    ]
    if manquantes:
        return 1
    if recus is None:
        recus = []

    prov, usage, defauts = lignes_du_tableau(fiches, extraits, recus)

    if defauts:
        return 1

    texte = tableau_provenance(prov) + "\n\n" + tableau_usage(usage) + "\n"
    if ecrire:
        io.open(CIBLE, "w", encoding="utf-8", newline="\n").write(texte)
    return 0


def main(argv):
    if "--construire" in argv:
        return construire()
    if "--essai" in argv:
        return construire(ecrire=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
