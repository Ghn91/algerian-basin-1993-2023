"""ERA5 monthly fields used in the revised analysis (surface wind stress,
mean sea-level pressure at the WeMO stations), with the request checked
against the CDS catalogue before submission and the received file checked
against the request.

Usage: python download_era5.py --schema | --telecharger | --verifier
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DONNEES = os.path.join(DOSSIER, "donnees")
SORTIES = os.path.join(DOSSIER, "sorties")

JEU = "reanalysis-era5-single-levels-monthly-means"
CATALOGUE = "https://cds.climate.copernicus.eu/api/catalogue/v1/collections"
DOI = "10.24381/cds.f17050d7"
LICENCE = "CC-BY-4.0"

ANNEE_DEBUT, ANNEE_FIN = 1993, 2023

REQUETES = {
    "instantane": {
        "variables": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
        ],
        "fichier": "era5_vent_msl_mensuel_1993_2023.nc",
        "attendues": ("u10", "v10", "msl"),
        "tensions_attendues": 0,
    },
    "wemo": {
        "variables": ["mean_sea_level_pressure"],
        "fichier": "era5_msl_poles_wemo_1993_2023.nc",
        "attendues": ("msl",),
        "tensions_attendues": 0,
        "emprise": [46.5, -7.5, 35.5, 13.0],
        "stations": {"San Fernando": (36.46, -6.20), "Padoue": (45.41, 11.88)},
    },
    "tension": {
        "variables": [
            "mean_eastward_turbulent_surface_stress",
            "mean_northward_turbulent_surface_stress",
        ],
        "fichier": "era5_tension_mensuelle_1993_2023.nc",
        "attendues": (),
        "tensions_attendues": 2,
    },
}

VARIABLES = [v for r in REQUETES.values() for v in r["variables"]]

DOMAINE_ARTICLE = {"ouest": -4.88, "est": 9.88, "sud": 35.12, "nord": 40.88}

MARGE_DEG = 2.0

EMPRISE = [
    DOMAINE_ARTICLE["nord"] + MARGE_DEG,
    DOMAINE_ARTICLE["ouest"] - MARGE_DEG,
    DOMAINE_ARTICLE["sud"] - MARGE_DEG,
    DOMAINE_ARTICLE["est"] + MARGE_DEG,
]

PAS_GRILLE = 0.25

NETCDF = os.path.join(DONNEES, "era5_vent_tension_msl_mensuel_1993_2023.nc")
ARCHIVE = os.path.join(SORTIES, "era5_request_schema.json")


def requete(annee_debut=ANNEE_DEBUT, annee_fin=ANNEE_FIN, emprise=None, variables=None):
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": list(variables if variables is not None else VARIABLES),
        "year": [str(a) for a in range(annee_debut, annee_fin + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": list(emprise if emprise is not None else EMPRISE),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _valeurs_du_widget(widget):
    det = widget.get("details") or {}
    if isinstance(det.get("values"), list):
        return set(det["values"])
    valeurs = set()
    for groupe in det.get("groups") or []:
        for v in groupe.get("values") or []:
            valeurs.add(v)
    return valeurs


def lire_schema(jeu=JEU, delai=40):
    try:
        brut = urllib.request.urlopen(
            f"{CATALOGUE}/{jeu}/form.json", timeout=delai
        ).read()
        widgets = {w.get("name"): w for w in json.loads(brut)}
        brut = urllib.request.urlopen(f"{CATALOGUE}/{jeu}", timeout=delai).read()
        return widgets, json.loads(brut)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None, None


def _dans_360(x):
    return x % 360.0


def _intervalle_longitudes(ouest, est):
    return _dans_360(ouest) <= 360.0 and _dans_360(est) <= 360.0


def confronter(req, widgets, collection):
    defauts = []
    if not widgets:
        return ["schema indisponible : la requete n'a pas ete confrontee"]

    for champ in (
        "product_type",
        "variable",
        "year",
        "month",
        "time",
        "data_format",
        "download_format",
    ):
        if champ not in req:
            continue
        admises = _valeurs_du_widget(widgets.get(champ, {}))
        if not admises:
            continue
        demandees = req[champ]
        if isinstance(demandees, str):
            demandees = [demandees]
        for v in demandees:
            if v not in admises:
                defauts.append(
                    f"{champ} : « {v} » n'est pas dans les "
                    f"{len(admises)} valeurs admises"
                )

    aire = req.get("area")
    if aire is not None:
        nord, ouest, sud, est = aire
        bbox = (
            ((collection or {}).get("extent") or {}).get("spatial", {}).get("bbox")
            or [[None] * 4]
        )[0]
        if None not in bbox:
            _, sud_min, _, nord_max = bbox
            if not (sud_min <= sud <= nord <= nord_max):
                defauts.append(
                    f"emprise : latitudes {sud}..{nord} hors de "
                    f"{sud_min}..{nord_max}"
                )
            if not _intervalle_longitudes(ouest, est):
                defauts.append(
                    f"emprise : longitudes {ouest}..{est} hors "
                    f"de l'emprise declaree"
                )
        if nord <= sud:
            defauts.append(
                f"emprise : le nord ({nord}) n'est pas au nord " f"du sud ({sud})"
            )

    annees = [int(a) for a in req.get("year", [])]
    interval = (
        ((collection or {}).get("extent") or {}).get("temporal", {}).get("interval")
        or [[None, None]]
    )[0]
    if annees and interval[0]:
        premier = int(str(interval[0])[:4])
        dernier = int(str(interval[1])[:4]) if interval[1] else date.today().year
        if min(annees) < premier or max(annees) > dernier:
            defauts.append(
                f"periode : {min(annees)}-{max(annees)} deborde "
                f"l'extension declaree {premier}-{dernier}"
            )
    return defauts


def marge_autour_du_domaine(emprise, domaine, pas=PAS_GRILLE):
    nord, ouest, sud, est = emprise
    return {
        "ouest": (domaine["ouest"] - ouest) / pas,
        "est": (est - domaine["est"]) / pas,
        "sud": (domaine["sud"] - sud) / pas,
        "nord": (nord - domaine["nord"]) / pas,
    }


def emprise_suffisante(emprise, domaine, mailles_min=1, pas=PAS_GRILLE):
    m = marge_autour_du_domaine(emprise, domaine, pas)
    return all(v >= mailles_min for v in m.values())


def stations_hors_emprise(emprise, stations, mailles_min=1, pas=PAS_GRILLE):
    nord, ouest, sud, est = emprise
    marge = mailles_min * pas
    dehors = []
    for nom, (lat, lon) in (stations or {}).items():
        if not (
            sud + marge <= lat <= nord - marge and ouest + marge <= lon <= est - marge
        ):
            dehors.append(f"{nom} ({lat} N, {lon} E)")
    return dehors


def mois_attendus(annee_debut=ANNEE_DEBUT, annee_fin=ANNEE_FIN):
    return (annee_fin - annee_debut + 1) * 12


def pas_de_grille(valeurs):
    import numpy as np

    d = np.abs(np.diff(np.asarray(valeurs, dtype=float)))
    if d.size == 0:
        return float("nan"), float("nan")
    return float(d.mean()), float(d.max() - d.min())


def marge_livree(lats, lons, domaine=None, pas=PAS_GRILLE):
    import numpy as np

    d = domaine if domaine is not None else DOMAINE_ARTICLE
    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    return {
        "ouest": (d["ouest"] - lon.min()) / pas,
        "est": (lon.max() - d["est"]) / pas,
        "sud": (d["sud"] - lat.min()) / pas,
        "nord": (lat.max() - d["nord"]) / pas,
    }


def stations_hors_grille(lats, lons, stations, mailles_min=1, pas=PAS_GRILLE):
    import numpy as np

    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    marge = mailles_min * pas
    dehors = []
    for nom, (la, lo) in (stations or {}).items():
        if not (
            lat.min() + marge <= la <= lat.max() - marge
            and lon.min() + marge <= lo <= lon.max() - marge
        ):
            dehors.append(
                f"{nom} ({la} N, {lo} E) — grille livree "
                f"[{lat.min():.2f};{lat.max():.2f}] N "
                f"[{lon.min():.2f};{lon.max():.2f}] E"
            )
    return dehors


def marge_des_stations(lats, lons, stations, pas=PAS_GRILLE):
    import numpy as np

    lat = np.asarray(lats, dtype=float)
    lon = np.asarray(lons, dtype=float)
    return {
        nom: min(la - lat.min(), lat.max() - la, lo - lon.min(), lon.max() - lo) / pas
        for nom, (la, lo) in (stations or {}).items()
    }


def signature(chemin, octets=4):
    with open(chemin, "rb") as f:
        return f.read(octets)


def est_une_archive(chemin):
    return signature(chemin) == b"PK\x03\x04"


def defauts_du_fichier(
    chemin=None,
    domaine=None,
    attendues=("u10", "v10", "msl"),
    tensions_attendues=0,
    stations=None,
):
    import xarray as xr

    chemin = chemin or NETCDF
    d = domaine if domaine is not None else DOMAINE_ARTICLE
    if not os.path.exists(chemin):
        return [f"fichier absent : {chemin}"], None

    if est_une_archive(chemin):
        noms = []
        try:
            import zipfile

            noms = zipfile.ZipFile(chemin).namelist()
        except Exception:
            pass
        return (
            [
                f"ARCHIVE ZIP portant l'extension .nc — "
                f"{len(noms)} fichier(s) dedans : {', '.join(noms)}. "
                f"Le CDS separe les variables par NATURE ; une requete qui en "
                f"mele deux rend une archive, non un fichier."
            ],
            None,
        )

    try:
        ds = xr.open_dataset(chemin)
    except Exception as e:
        return [f"illisible par xarray : {type(e).__name__} — {e}"], None

    try:
        return _confronter_le_contenu(
            ds, chemin, d, attendues, tensions_attendues, stations
        )
    finally:
        ds.close()


def _confronter_le_contenu(ds, chemin, d, attendues, tensions_attendues, stations=None):
    import numpy as np

    defauts = []

    manquantes = [v for v in attendues if v not in ds.data_vars]
    if manquantes:
        defauts.append(f"variables absentes : {', '.join(manquantes)}")

    tensions = [
        v
        for v in ds.data_vars
        if "N m**-2" in str(ds[v].attrs.get("units", ""))
        or "N m-2" in str(ds[v].attrs.get("units", ""))
    ]
    if len(tensions) < tensions_attendues:
        defauts.append(
            f"surface stress: {len(tensions)} variable(s) in N m-2 instead of {tensions_attendues}; the Ekman index requires them, stress cannot be computed from the mean wind"
        )

    nom_t = next((n for n in ("valid_time", "time") if n in ds.coords), None)
    if nom_t is None:
        defauts.append("aucun axe de temps reconnu")
    else:
        t = ds[nom_t].values
        if len(t) != mois_attendus():
            defauts.append(
                f"temps : {len(t)} pas pour " f"{mois_attendus()} mois attendus"
            )
        ecarts = np.diff(t).astype("timedelta64[D]").astype(int)
        if ecarts.size and (ecarts.min() < 28 or ecarts.max() > 31):
            defauts.append(
                f"temps : ecarts de {ecarts.min()} a "
                f"{ecarts.max()} jours — les pas ne sont pas"
                f" mensuels"
            )

    for axe in ("latitude", "longitude"):
        if axe not in ds.coords:
            defauts.append(f"axe absent : {axe}")
            continue
        moyen, etendue = pas_de_grille(ds[axe].values)
        if abs(moyen - PAS_GRILLE) > 1e-6 or etendue > 1e-6:
            defauts.append(
                f"{axe} : pas {moyen:.4f}° (etendue {etendue:.1e})"
                f" au lieu de {PAS_GRILLE}°"
            )

    if "latitude" in ds.coords and "longitude" in ds.coords:
        if stations:
            dehors = stations_hors_grille(
                ds.latitude.values, ds.longitude.values, stations
            )
            if dehors:
                defauts.append(
                    "stations hors de la grille livree : " + " ; ".join(dehors)
                )
        else:
            m = marge_livree(ds.latitude.values, ds.longitude.values, d)
            if any(v < 1.0 for v in m.values()):
                defauts.append(
                    "marge : moins d'une maille autour du domaine — "
                    + " ".join(f"{k}:{v:.2f}" for k, v in m.items())
                )

    if "expver" in ds.coords:
        versions = np.unique(np.atleast_1d(ds["expver"].values))
        if len(versions) > 1:
            defauts.append(
                f"expver : {len(versions)} versions melangees "
                f"({versions}) — ERA5 et ERA5T ne sont pas le meme"
                f" produit"
            )

    for v in ds.data_vars:
        part = float(np.isnan(ds[v].values).mean())
        if part > 0:
            defauts.append(f"{v} : {part:.4%} de valeurs manquantes")

    resume = {"variables": list(ds.data_vars)}
    if nom_t is not None:
        t = ds[nom_t].values
        resume["n_pas"] = len(t)
        resume["debut"] = str(np.datetime_as_string(t[0], "D"))
        resume["fin"] = str(np.datetime_as_string(t[-1], "D"))
    if "latitude" in ds.coords and "longitude" in ds.coords:
        resume["n_lat"] = int(ds.latitude.size)
        resume["n_lon"] = int(ds.longitude.size)
        if stations:
            resume["stations"] = marge_des_stations(
                ds.latitude.values, ds.longitude.values, stations
            )
        else:
            resume["marge"] = marge_livree(ds.latitude.values, ds.longitude.values, d)
    if "expver" in ds.coords:
        resume["expver"] = [
            str(x) for x in np.unique(np.atleast_1d(ds["expver"].values))
        ]
    resume["unites"] = {v: str(ds[v].attrs.get("units", "?")) for v in ds.data_vars}
    return defauts, resume


def montrer_verification(
    chemin=None, attendues=("u10", "v10", "msl"), tensions_attendues=0, stations=None
):
    defauts, resume = defauts_du_fichier(
        chemin,
        attendues=attendues,
        tensions_attendues=tensions_attendues,
        stations=stations,
    )
    chemin = chemin or NETCDF
    return defauts


def verifier_tout():
    total = []
    for nom, r in REQUETES.items():
        total += montrer_verification(
            os.path.join(DONNEES, r["fichier"]),
            attendues=r["attendues"],
            tensions_attendues=r["tensions_attendues"],
            stations=r.get("stations"),
        )
    return total


def montrer_schema(req=None):
    req = req if req is not None else requete()
    widgets, collection = lire_schema()
    defauts = confronter(req, widgets, collection)
    os.makedirs(SORTIES, exist_ok=True)
    with open(ARCHIVE, "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "releve_le": date.today().isoformat(),
                "jeu": JEU,
                "doi": DOI,
                "requete": req,
                "defauts": defauts,
                "widgets_lus": sorted(widgets) if widgets else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return defauts


def telecharger():
    for nom, r in REQUETES.items():
        cible = os.path.join(DONNEES, r["fichier"])
        req = requete(variables=r["variables"], emprise=r.get("emprise"))
        defauts = montrer_schema(req)
        if defauts:
            return 1
        if r.get("stations"):
            dehors = stations_hors_emprise(req["area"], r["stations"])
            if dehors:
                return 1
        elif not emprise_suffisante(req["area"], DOMAINE_ARTICLE):
            return 1
        if os.path.exists(cible):
            continue
        try:
            import cdsapi
        except ImportError:
            return 2
        os.makedirs(DONNEES, exist_ok=True)
        cdsapi.Client().retrieve(JEU, req, cible)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--schema",
        action="store_true",
        help="check the request against the catalogue without submitting it",
    )
    p.add_argument("--telecharger", action="store_true")
    p.add_argument(
        "--verifier",
        action="store_true",
        help="check the received file against the request",
    )
    a = p.parse_args()
    if a.schema:
        montrer_schema()
        return 0
    if a.telecharger:
        code = telecharger()
        if code != 0:
            return code
        return 1 if verifier_tout() else 0
    if a.verifier:
        return 1 if verifier_tout() else 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
