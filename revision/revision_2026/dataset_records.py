"""Records of the datasets used, read from the public Copernicus catalogues
(product identifier, version, provider, DOI) and from the coordinates of the
files actually received (grid, period, variables, depth).

Usage: python dataset_records.py --relever | --recevoir [--apercu DIR]
"""

import csv
import io
import json
import os
import re
import sys
import time

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
REVISE = os.path.join(DOSSIER, "manuscript", "main.tex")
SORTIES = os.path.join(DOSSIER, "sorties")
RACINE_STAC = "https://stac.marine.copernicus.eu/metadata"

EXTRACTIONS_SOURCES = {
    "depot public": os.path.join(
        os.path.dirname(RACINE),
        "algerian-basin-1993-2023",
        "src",
        "00_download_data.py",
    ),
    "altimetrie, ce dossier": os.path.join(DOSSIER, "download_altimetry.py"),
    "Gibraltar, ce dossier": os.path.join(DOSSIER, "gibraltar_inflow.py"),
}
EXTRACTION = EXTRACTIONS_SOURCES["depot public"]

RACINE_DONNEES = os.environ.get("ARTICLE_DONNEES") or os.path.join(
    os.path.expanduser("~"), "donnees_article"
)
DOSSIERS_DONNEES = {
    "disque, hors depot": RACINE_DONNEES,
    "altimetrie, ce dossier": os.path.join(DOSSIER, "donnees"),
}

FICHIERS_DECLARES = {
    "ERA5_MSLP_1993_2023_monthly.nc": {
        "jeu": "reanalysis-era5-single-levels-monthly-means",
        "raison": "telechargement de mars 2026 (figure 8 de la version soumise) ; emprise -10 a 20 E / 30 a 50 N",
    },
}

COORDONNEES = {
    "longitude": ("longitude", "lon", "x"),
    "latitude": ("latitude", "lat", "y"),
    "temps": ("time", "time_counter", "valid_time"),
    "profondeur": ("depth", "deptht", "lev", "z"),
}

MOTIF_PRODUIT = re.compile(r"[A-Z][A-Z0-9]*(?:\\_[A-Z0-9]+){3,}")

TENTATIVES = 3
ATTENTE = 4.0

HORS_CATALOGUE_CMEMS = {
    "TWO-SAT": {
        "sonde": "10.24381/cds.4c328c78",
        "jeu": "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        "fournisseur": "Copernicus Climate Change Service (C3S)",
        "raison": "produit du CDS, non du Marine Service : le catalogue STAC "
        "de CMEMS ne le porte pas",
        "collection": "satellite-sea-level-global",
        "titre": "Sea level gridded data from satellite observations for "
        "the global ocean from 1993 to present",
        "doi": "10.24381/cds.4c328c78",
        "licence": "CC-BY-4.0",
        "periode": "1993-01-01 -> 2026-01-16",
        "a_faire": "DOI publie par le CDS pour cette collection, publiee en 2018 et couvrant toutes ses variantes",
    },
}


def produits_du_manuscrit(texte):
    vus, out = set(), []
    for m in MOTIF_PRODUIT.finditer(texte):
        i = m.group(0).replace("\\_", "_")
        if i not in vus:
            vus.add(i)
            out.append(i)
    return out


def extractions(source, methode="subset", rendre_sautes=False):
    import ast

    out, sautes = [], []
    try:
        arbre = ast.parse(source)
    except SyntaxError:
        return out
    for n in ast.walk(arbre):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        nom = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if nom != methode:
            continue
        kw = {}
        for k in n.keywords:
            if k.arg is None:
                continue
            try:
                kw[k.arg] = ast.literal_eval(k.value)
            except (ValueError, SyntaxError):
                kw[k.arg] = None
        if not kw.get("dataset_id"):
            sautes.append(ast.unparse(f) if hasattr(ast, "unparse") else "?")
            continue
        v = kw.get("variables") or []
        dmin, dmax = kw.get("minimum_depth"), kw.get("maximum_depth")
        nus = {
            "lon_min_E": kw.get("minimum_longitude"),
            "lon_max_E": kw.get("maximum_longitude"),
            "lat_min_N": kw.get("minimum_latitude"),
            "lat_max_N": kw.get("maximum_latitude"),
        }
        out.append(
            {
                "jeu": kw["dataset_id"],
                "variables_extraites": " ; ".join(v) if isinstance(v, list) else "",
                "variables_portees_declarees": (
                    " ; ".join(v) if isinstance(v, list) else ""
                ),
                "emprise": (
                    (
                        f"{kw.get('minimum_longitude')} a "
                        f"{kw.get('maximum_longitude')} E, "
                        f"{kw.get('minimum_latitude')} a "
                        f"{kw.get('maximum_latitude')} N"
                    )
                    if kw.get("minimum_longitude") is not None
                    else ""
                ),
                "periode_extraite": (
                    (
                        f"{str(kw.get('start_datetime'))[:10]} -> "
                        f"{str(kw.get('end_datetime'))[:10]}"
                    )
                    if kw.get("start_datetime")
                    else ""
                ),
                "debut_extrait": str(kw.get("start_datetime") or "")[:10],
                "fin_extrait": str(kw.get("end_datetime") or "")[:10],
                "lon_min_E": "" if nus["lon_min_E"] is None else f"{nus['lon_min_E']}",
                "lon_max_E": "" if nus["lon_max_E"] is None else f"{nus['lon_max_E']}",
                "lat_min_N": "" if nus["lat_min_N"] is None else f"{nus['lat_min_N']}",
                "lat_max_N": "" if nus["lat_max_N"] is None else f"{nus['lat_max_N']}",
                "profondeur_min_m": "" if dmin is None else f"{dmin}",
                "profondeur_max_m": "" if dmax is None else f"{dmax}",
                "profondeur_m": (
                    ""
                    if dmin is None
                    else (f"{dmin}" if dmin == dmax else f"{dmin} a {dmax}")
                ),
            }
        )
    if rendre_sautes:
        return out, len(sautes)
    return out


def extractions_cds(module=None):
    m = module
    if m is None:
        import importlib.util

        chemin = os.path.join(DOSSIER, "download_era5.py")
        if not os.path.isfile(chemin):
            return []
        s = importlib.util.spec_from_file_location("_c07_lu", chemin)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    out = []
    for nom, r in sorted(getattr(m, "REQUETES", {}).items()):
        e = r.get("emprise") or getattr(m, "EMPRISE", None)
        nord, ouest, sud, est = e if e and len(e) == 4 else (None, None, None, None)
        out.append(
            {
                "requete": nom,
                "jeu": getattr(m, "JEU", ""),
                "doi": getattr(m, "DOI", ""),
                "licence": getattr(m, "LICENCE", ""),
                "fichier": r.get("fichier", ""),
                "variables_extraites": " ; ".join(r.get("variables") or []),
                "variables_portees_declarees": " ; ".join(r.get("attendues") or []),
                "lon_min_E": "" if ouest is None else f"{ouest}",
                "lon_max_E": "" if est is None else f"{est}",
                "lat_min_N": "" if sud is None else f"{sud}",
                "lat_max_N": "" if nord is None else f"{nord}",
                "debut_extrait": f"{getattr(m, 'ANNEE_DEBUT', '')}-01-01",
                "fin_extrait": f"{getattr(m, 'ANNEE_FIN', '')}-12-31",
                "profondeur_min_m": "",
                "profondeur_max_m": "",
                "source": "CDS, ce dossier",
            }
        )
    return out


def bornes(coords, declarees=None):
    d = COORDONNEES if declarees is None else declarees
    axes, ecartees = {}, []
    for nom, valeurs in coords.items():
        axe = None
        for a, synonymes in d.items():
            if nom.lower() in synonymes:
                axe = a
                break
        if axe is None:
            ecartees.append((nom, "aucun axe declare"))
            continue
        if axe in axes:
            ecartees.append(
                (
                    nom,
                    f"double emploi de l'axe {axe}, "
                    f"deja pris par {axes[axe]['nom']}",
                )
            )
            continue
        v = list(valeurs)
        axes[axe] = {
            "nom": nom,
            "n": len(v),
            "min": min(v) if v else None,
            "max": max(v) if v else None,
        }
    return axes, ecartees


def jeu_du_nom(nom, jeux):
    base = os.path.basename(nom)
    candidats = [j for j in jeux if j and base.startswith(j)]
    return max(candidats, key=len) if candidats else None


def accord(appel, recu):
    details = []
    confronte = False

    def pas(a):
        if not a or a["n"] < 2 or a["min"] is None:
            return None
        return (a["max"] - a["min"]) / (a["n"] - 1)

    for axe, (dmin, dmax) in (
        ("longitude", ("lon_min", "lon_max")),
        ("latitude", ("lat_min", "lat_max")),
    ):
        amin, amax = appel.get(dmin), appel.get(dmax)
        a = (recu.get("axes") or {}).get(axe)
        if amin is None or amax is None or not a or a["min"] is None:
            continue
        confronte = True
        p = pas(a) or 0.0
        if a["min"] < amin - 1e-9 or a["max"] > amax + 1e-9:
            details.append(
                f"{axe} : le fichier sort du demande "
                f"({a['min']} a {a['max']} contre {amin} a {amax})"
            )
        elif p and (a["min"] - amin > p + 1e-9 or amax - a["max"] > p + 1e-9):
            details.append(
                f"{axe} : le fichier est en retrait de plus d'une "
                f"maille ({a['min']} a {a['max']} contre "
                f"{amin} a {amax}, maille {p})"
            )

    if appel.get("debut") and appel.get("fin"):
        a = (recu.get("axes") or {}).get("temps")
        if a and a["min"] is not None:
            confronte = True
            d, f = str(a["min"])[:10], str(a["max"])[:10]
            if d < appel["debut"][:10] or f > appel["fin"][:10]:
                details.append(
                    f"temps : le fichier sort du demande "
                    f"({d} -> {f} contre {appel['debut'][:10]} -> "
                    f"{appel['fin'][:10]})"
                )

    dem = [v.strip() for v in (appel.get("variables") or "").split(";") if v.strip()]
    if dem and recu.get("variables_portees"):
        confronte = True
        porte = {v.strip() for v in recu["variables_portees"].split(";")}
        manque = [v for v in dem if v not in porte]
        if manque:
            details.append(
                "variables demandees et absentes du fichier : " + ", ".join(manque)
            )

    pmin, pmax = appel.get("prof_min"), appel.get("prof_max")
    a = (recu.get("axes") or {}).get("profondeur")
    if pmin is not None and a and a["min"] is not None:
        confronte = True
        if (
            abs(a["min"] - pmin) > 1e-6
            or abs(a["max"] - (pmax if pmax is not None else pmin)) > 1e-6
        ):
            details.append(
                f"profondeur : {a['min']} a {a['max']} dans le "
                f"fichier contre {pmin} a {pmax} demandes"
            )

    if not confronte:
        return "rien a confronter", details
    return ("accord" if not details else "DESACCORD"), details


def etiquettes_du_manuscrit(texte, declarees=None):
    d = HORS_CATALOGUE_CMEMS if declarees is None else declarees
    out = {}
    for e, x in d.items():
        s = (x or {}).get("sonde") or e
        out[e] = (s in texte) or (s.replace("_", "\\_") in texte)
    return out


def resolution(propr):
    g = (propr or {}).get("geoResolution") or {}
    r, c = g.get("row") or {}, g.get("column") or {}

    def dire(d):
        if d.get("magnitude") is None:
            return None
        return f"{d['magnitude']} {d.get('units', '')}".strip()

    a, b = dire(r), dire(c)
    if a and b and a != b:
        return f"{a} x {b}"
    return a or b


def periode(collection):
    t = (collection.get("extent") or {}).get("temporal") or {}
    iv = (t.get("interval") or [[None, None]])[0]

    def jour(x):
        return (x or "").split("T")[0] or ""

    return jour(iv[0]), jour(iv[1] if len(iv) > 1 else None)


def jeux_du_produit(collection):
    out = []
    for l in collection.get("links") or []:
        if l.get("rel") != "item":
            continue
        h = l.get("href") or ""
        i = h.split("/")[0] if "/" in h else h.replace(".stac.json", "")
        if i:
            out.append((i, l.get("title") or ""))
    return out


def fiche(collection):
    p = collection.get("properties") or {}
    d, f = periode(collection)
    fournisseurs = [
        x.get("name", "")
        for x in (collection.get("providers") or [])
        if "producer" in (x.get("roles") or [])
    ]
    return {
        "produit": collection.get("id") or "",
        "titre": collection.get("title") or "",
        "producteur": " ; ".join(fournisseurs),
        "doi": collection.get("sci:doi") or "",
        "niveau_de_traitement": p.get("processingLevel") or "",
        "resolution_spatiale": resolution(p) or "",
        "resolutions_temporelles": " ; ".join(p.get("tempResolutions") or []),
        "periode_debut": d,
        "periode_fin": f or "(en cours de production)",
        "niveaux_verticaux": (
            "" if p.get("vertLevels") is None else str(p["vertLevels"])
        ),
        "extension_verticale_m": (
            ""
            if p.get("vertExtentMin") is None
            else f"{p.get('vertExtentMax')} a {p.get('vertExtentMin')}"
        ),
        "variables_principales": " ; ".join(p.get("mainVariables") or []),
        "variables_toutes": " ; ".join(p.get("allVariables") or []),
        "jeux": " ; ".join(i for i, _ in jeux_du_produit(collection)),
    }


def collection(produit, racine=RACINE_STAC, tentatives=TENTATIVES):
    from urllib.request import urlopen

    url = f"{racine}/{produit}/product.stac.json"
    for n in range(tentatives):
        try:
            with urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            if n + 1 < tentatives:
                time.sleep(ATTENTE)
    return None


def pas_declares(produit, jeu, racine=RACINE_STAC, tentatives=TENTATIVES):
    from urllib.request import urlopen

    url = f"{racine}/{produit}/{jeu}/dataset.stac.json"
    for n in range(tentatives):
        try:
            with urlopen(url, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
            return pas_des_dimensions(
                (d.get("properties") or {}).get("cube:dimensions")
            )
        except Exception:
            if n + 1 < tentatives:
                time.sleep(ATTENTE)
    return None


def relever():
    if not os.path.isfile(REVISE):
        return 1
    produits = produits_du_manuscrit(io.open(REVISE, encoding="utf-8").read())
    if not produits:
        return 1

    texte = io.open(REVISE, encoding="utf-8").read()
    etiq = etiquettes_du_manuscrit(texte)
    lignes, archive, injoignables = [], {}, []
    for i in produits:
        c = collection(i)
        if c is None:
            injoignables.append(i)
            continue
        f = fiche(c)
        f["etat"] = "releve au catalogue CMEMS"
        archive[i] = f
        lignes.append(f)

    orphelines = []
    for e, x in sorted(HORS_CATALOGUE_CMEMS.items()):
        trouvee = etiq.get(e, False)
        if not trouvee:
            orphelines.append(e)
        f = {k: "" for k in (lignes[0] if lignes else {})}
        f.update(
            {
                "produit": e,
                "titre": f"[{e}]",
                "producteur": x["fournisseur"],
                "jeux": x["jeu"],
                "doi": x.get("doi", ""),
                "etat": (
                    "declare hors catalogue CMEMS, a relever"
                    if trouvee
                    else "declaration orpheline — le manuscrit ne le " "nomme plus"
                ),
            }
        )
        lignes.append(f)
        archive[e] = dict(
            f, raison=x["raison"], a_faire=x["a_faire"], sonde=x.get("sonde") or e
        )

    cds = extractions_cds()
    collections = {}
    for x in cds:
        collections.setdefault(x["jeu"], x)
    for j, x in sorted(collections.items()):
        sonde = x.get("doi") or j
        trouvee = (sonde in texte) or (j in texte)
        if not trouvee:
            orphelines.append(j)
        f = {k: "" for k in (lignes[0] if lignes else {})}
        f.update(
            {
                "produit": j,
                "titre": f"[{j}]",
                "producteur": "Copernicus Climate Change Service (C3S)",
                "doi": x.get("doi", ""),
                "jeux": j,
                "resolutions_temporelles": "Monthly",
                "etat": (
                    "declare hors catalogue CMEMS, releve au CDS"
                    if trouvee
                    else "declaration orpheline — le manuscrit ne porte "
                    "ni la collection ni son DOI"
                ),
            }
        )
        lignes.append(f)
        archive[j] = dict(
            f,
            sonde=sonde,
            raison="collection du Climate Data Store, non du " "Marine Service",
            a_faire="collection et DOI portes au manuscrit",
        )

    os.makedirs(SORTIES, exist_ok=True)
    ext, sautes_total, absents = [], 0, []
    for nom, chemin in EXTRACTIONS_SOURCES.items():
        if not os.path.isfile(chemin):
            absents.append(nom)
            continue
        e, saut = extractions(
            io.open(chemin, encoding="utf-8", errors="replace").read(),
            rendre_sautes=True,
        )
        for x in e:
            x["source"] = nom
        ext.extend(e)
        sautes_total += saut
    if ext:
        ce = os.path.join(SORTIES, "datasets_extraction.csv")
        with io.open(ce, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ext[0]))
            w.writeheader()
            w.writerows(ext)

    c = os.path.join(SORTIES, "datasets_records.csv")
    if lignes:
        with io.open(c, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
            w.writeheader()
            w.writerows(lignes)

    j = os.path.join(SORTIES, "datasets_catalogue.json")
    io.open(j, "w", encoding="utf-8", newline="\n").write(
        json.dumps(
            {
                "produits_lus_au_manuscrit": produits,
                "injoignables": injoignables,
                "hors_catalogue_declares": sorted(HORS_CATALOGUE_CMEMS),
                "declarations_orphelines": orphelines,
                "extraction_sources": {
                    k: os.path.isfile(v) for k, v in EXTRACTIONS_SOURCES.items()
                },
                "extraction_appels_sautes": sautes_total,
                "extractions": ext,
                "fiches": archive,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    if orphelines:
        return 1
    if injoignables:
        return 1
    return 0


def ouvrir_netcdf(chemin):
    import xarray as xr

    ds = xr.open_dataset(chemin, decode_timedelta=False)
    try:
        return ({c: ds[c].values for c in ds.coords}, list(ds.data_vars))
    finally:
        ds.close()


def pas_des_dimensions(dims):
    out = {}
    for k, v in (dims or {}).items():
        if not isinstance(v, dict):
            continue
        a, s = v.get("axis"), v.get("step")
        if a in ("x", "y") and isinstance(s, (int, float)):
            out["lon" if a == "x" else "lat"] = float(s)
    return out


def pas_etendue(valeurs):
    if valeurs is None:
        return ""
    v = list(valeurs)
    if len(v) < 2:
        return ""
    try:
        a, b = float(min(v)), float(max(v))
        return f"{(b - a) / (len(v) - 1):.9f}"
    except Exception:
        return ""


def pas_median(valeurs):
    if valeurs is None:
        return ""
    v = list(valeurs)
    if len(v) < 2:
        return ""
    try:
        import numpy as np

        d = np.abs(np.diff(np.asarray(v, dtype="float64")))
        return f"{float(np.median(d)):.6f}"
    except Exception:
        return ""


def cadence_jours(valeurs):
    if valeurs is None:
        return ""
    v = list(valeurs)
    if len(v) < 2:
        return ""
    try:
        import numpy as np

        t = np.asarray(v)
        d = np.diff(t)
        if np.issubdtype(t.dtype, np.datetime64):
            j = d.astype("timedelta64[s]").astype(float) / 86400.0
        else:
            j = d.astype(float)
        return f"{float(np.median(j)):.4f}"
    except Exception:
        return ""


def fiche_du_fichier(chemin, jeux, ouvrir=None):
    coords, variables = (ouvrir or ouvrir_netcdf)(chemin)
    axes, ecartees = bornes(coords)
    lo, la = axes.get("longitude"), axes.get("latitude")
    t, p = axes.get("temps"), axes.get("profondeur")
    base = os.path.basename(chemin)
    declare = FICHIERS_DECLARES.get(base, {})
    return {
        "fichier": base,
        "jeu": jeu_du_nom(chemin, jeux) or declare.get("jeu", ""),
        "declare": declare.get("raison", ""),
        "doi": "",
        "licence": "",
        "octets": os.path.getsize(chemin) if os.path.isfile(chemin) else "",
        "variables_portees": " ; ".join(sorted(variables)),
        "emprise_portee": (
            (f"{lo['min']} a {lo['max']} E, " f"{la['min']} a {la['max']} N")
            if lo and la and lo["min"] is not None
            else ""
        ),
        "mailles_lon": lo["n"] if lo else "",
        "mailles_lat": la["n"] if la else "",
        "pas_lon_deg": (
            pas_etendue(coords.get(lo["nom"])) if lo and lo["nom"] in coords else ""
        ),
        "pas_lat_deg": (
            pas_etendue(coords.get(la["nom"])) if la and la["nom"] in coords else ""
        ),
        "pas_lon_median": (
            pas_median(coords.get(lo["nom"])) if lo and lo["nom"] in coords else ""
        ),
        "pas_lat_median": (
            pas_median(coords.get(la["nom"])) if la and la["nom"] in coords else ""
        ),
        "periode_portee": (
            (f"{str(t['min'])[:10]} -> {str(t['max'])[:10]}")
            if t and t["min"] is not None
            else ""
        ),
        "pas_de_temps": t["n"] if t else "",
        "cadence_jours": (
            cadence_jours(coords.get(t["nom"])) if t and t["nom"] in coords else ""
        ),
        "profondeur_min_m": "" if not p or p["min"] is None else f"{p['min']}",
        "profondeur_max_m": "" if not p or p["max"] is None else f"{p['max']}",
        "niveaux": p["n"] if p else "",
        "coordonnees_ecartees": " ; ".join(f"{n} ({r})" for n, r in ecartees),
        "axes": axes,
    }


def sans_millesime(jeu):
    return re.sub(r"_\d{6}(?:--ext--[\w.-]+)?$", "", jeu.strip())


def jeux_connus(archive, extractions_lues):
    out = set()
    for e, x in HORS_CATALOGUE_CMEMS.items():
        out.add(x["jeu"])
    for k, f in (archive.get("fiches") or {}).items():
        for j in (f.get("jeux") or "").split(";"):
            if j.strip():
                out.add(j.strip())
                out.add(sans_millesime(j))
    for x in extractions_lues:
        if x.get("jeu"):
            out.add(x["jeu"])
    return {j for j in out if j}


def recevoir(apercu=None):
    a = os.path.join(SORTIES, "datasets_catalogue.json")
    if not os.path.isfile(a):
        return 1
    archive = json.loads(io.open(a, encoding="utf-8").read())

    ext = []
    for nom, chemin in EXTRACTIONS_SOURCES.items():
        if os.path.isfile(chemin):
            for x in extractions(
                io.open(chemin, encoding="utf-8", errors="replace").read()
            ):
                x["source"] = nom
                ext.append(x)
    cds = extractions_cds()
    jeux = jeux_connus(archive, ext) | {x["jeu"] for x in cds if x["jeu"]}
    par_jeu = {}
    for x in ext:
        par_jeu.setdefault(x["jeu"], x)
    par_fichier = {x["fichier"]: x for x in cds if x.get("fichier")}

    lignes, absents, illisibles, sans_jeu, desaccords = [], [], [], [], []
    declares, pas_en_defaut = {}, []
    for nom, dossier in DOSSIERS_DONNEES.items():
        if not os.path.isdir(dossier):
            absents.append((nom, dossier))
            continue
        trouves = sorted(f for f in os.listdir(dossier) if f.endswith(".nc"))
        for f in trouves:
            chemin = os.path.join(dossier, f)
            try:
                fi = fiche_du_fichier(chemin, jeux)
            except Exception as e:
                illisibles.append((f, f"{type(e).__name__}: {e}"))
                continue
            fi["source"] = nom
            app = par_fichier.get(fi["fichier"])
            if app:
                fi["jeu"] = app["jeu"]
                fi["doi"] = app.get("doi", "")
                fi["licence"] = app.get("licence", "")
            if not fi["jeu"]:
                sans_jeu.append(f)
            app = app or par_jeu.get(fi["jeu"])
            if app:
                e, det = accord(
                    {
                        "lon_min": _f(app.get("lon_min_E")),
                        "lon_max": _f(app.get("lon_max_E")),
                        "lat_min": _f(app.get("lat_min_N")),
                        "lat_max": _f(app.get("lat_max_N")),
                        "debut": app.get("debut_extrait") or None,
                        "fin": app.get("fin_extrait") or None,
                        "prof_min": _f(app.get("profondeur_min_m")),
                        "prof_max": _f(app.get("profondeur_max_m")),
                        "variables": app.get("variables_portees_declarees"),
                    },
                    fi,
                )
            else:
                e, det = "aucun appel archive", []
            fi["accord_avec_l_appel"] = e
            fi["desaccord_detail"] = " ; ".join(det)
            fi["variables_confrontees"] = (
                "oui"
                if (app or {}).get("variables_portees_declarees")
                else "non : le telechargeur ne declare pas les noms portes"
            )
            if e == "DESACCORD":
                desaccords.append(f)
            fi["pas_lon_declare"] = fi["pas_lat_declare"] = ""
            fi["pas_declares_accord"] = "aucun jeu connu"
            if fi["jeu"]:
                if fi["jeu"] not in declares:
                    p = next(
                        (
                            k
                            for k, f2 in (archive.get("fiches") or {}).items()
                            if fi["jeu"] in (f2.get("jeux") or "")
                        ),
                        "",
                    )
                    jv = next(
                        (
                            j.strip()
                            for j in (
                                (archive.get("fiches") or {}).get(p, {}).get("jeux")
                                or ""
                            ).split(";")
                            if j.strip().startswith(fi["jeu"])
                        ),
                        "",
                    )
                    declares[fi["jeu"]] = pas_declares(p, jv) if p and jv else None
                d = declares[fi["jeu"]]
                if d is None:
                    fi["pas_declares_accord"] = "non releve"
                elif not d:
                    fi["pas_declares_accord"] = "le jeu ne declare aucun pas " "spatial"
                else:
                    fi["pas_lon_declare"] = f"{d.get('lon', '')}"
                    fi["pas_lat_declare"] = f"{d.get('lat', '')}"
                    ecarts = []
                    for axe, col in (("lon", "pas_lon_deg"), ("lat", "pas_lat_deg")):
                        m, v = _f(fi.get(col)), d.get(axe)
                        if m is None or v is None:
                            continue
                        if abs(m - v) > 1e-6 * max(1.0, abs(v)):
                            ecarts.append(f"{axe} mesure {m} contre {v} " "declare")
                    fi["pas_declares_accord"] = (
                        "accord" if not ecarts else "DESACCORD : " + " ; ".join(ecarts)
                    )
                    if ecarts:
                        pas_en_defaut.append(fi["fichier"])
            fi.pop("axes")
            lignes.append(fi)

    vus_noms = {l["fichier"] for l in lignes}
    orphelins_declares = [f for f in FICHIERS_DECLARES if f not in vus_noms]

    if lignes:
        ou = apercu or SORTIES
        os.makedirs(ou, exist_ok=True)
        c = os.path.join(ou, "datasets_files_received.csv")
        with io.open(c, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lignes[0]))
            w.writeheader()
            w.writerows(lignes)

    if desaccords:
        for l in lignes:
            if l.get("accord_avec_l_appel") != "DESACCORD":
                continue
            manquantes = re.findall(
                r"ABSENTES du fichier : (.+?)(?:;|$)", l.get("desaccord_detail") or ""
            )
            if not manquantes:
                continue
            produit = ""
            for p, f in (archive.get("fiches") or {}).items():
                if l["jeu"] and l["jeu"] in (f.get("jeux") or ""):
                    produit = p
                    break
            if not produit:
                continue
        return 1
    if pas_en_defaut:
        return 1
    if orphelins_declares:
        return 1
    if illisibles or sans_jeu:
        return 1
    return 0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(argv):
    if "--relever" in argv:
        return relever()
    if "--recevoir" in argv:
        a = None
        if "--apercu" in argv:
            i = argv.index("--apercu")
            if i + 1 >= len(argv):
                return 1
            a = argv[i + 1]
        return recevoir(apercu=a)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
