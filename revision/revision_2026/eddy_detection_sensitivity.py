"""Sensitivity of the eddy catalogue to the filter cut-off (500, 1000 km)
and to the contour interval (1, 4 mm), on one day in ten, one setting changed
at a time; the reference setting must reproduce the catalogue of eddy_atlas.py.
Requires the environment of environment_detection.yml.

Usage: python eddy_detection_sensitivity.py --produire [--pas N] | --bilan
"""

import csv
import io
import os
import sys
import tempfile
from datetime import datetime

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
SORTIES = os.path.join(DOSSIER, "sorties")
ATLAS_REFERENCE = os.path.join(SORTIES, "eddy_atlas")
FICHIER_ALLSAT = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_"
    "multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)
SORTIE_JOURS = os.path.join(SORTIES, "detection_sensitivity_by_day.csv")
SORTIE_BILAN = os.path.join(SORTIES, "detection_sensitivity_summary.csv")

REFERENCE = {"filtre_km": 700, "contour_m": 0.002, "erreur_forme": 55}
VARIANTES = [
    ("reference", {}),
    ("filtre 500 km", {"filtre_km": 500}),
    ("filtre 1000 km", {"filtre_km": 1000}),
    ("contour 0.001 m", {"contour_m": 0.001}),
    ("contour 0.004 m", {"contour_m": 0.004}),
]
PAS_DEFAUT = 10


def jours_echantillon(n_jours, pas):
    if pas < 1:
        raise ValueError("step >= 1")
    return list(range(0, n_jours, pas))


def reglage(variante):
    r = dict(REFERENCE)
    r.update(variante)
    return r


def comparer_a_l_atlas(retenues, atlas):
    if len(retenues) != len(atlas):
        return f"{len(retenues)} retenues contre {len(atlas)} dans l'atlas"
    for a, b in zip(
        sorted(retenues, key=lambda r: (r["lon"], r["lat"])),
        sorted(atlas, key=lambda r: (r["lon"], r["lat"])),
    ):
        if abs(a["lon"] - b["lon"]) > 1e-4 or abs(a["lat"] - b["lat"]) > 1e-4:
            return f"position {a['lon']},{a['lat']} contre {b['lon']},{b['lat']}"
        if abs(a["amplitude"] - b["amplitude"]) > 1e-9:
            return f"amplitude {a['amplitude']} contre {b['amplitude']}"
    return None


def bilan(lignes):
    out = []
    for nom, _ in VARIANTES:
        for pol in ("anticyclone", "cyclone"):
            L = [l for l in lignes if l["variante"] == nom and l["polarite"] == pol]
            if not L:
                continue
            n = len(L)
            ident = sum(l["identifiees"] for l in L)
            ret = sum(l["retenues"] for l in L)
            amp = [l["amplitude_moyenne_m"] for l in L if l["retenues"] > 0]
            ray = [l["rayon_moyen_m"] for l in L if l["retenues"] > 0]
            pond = [l["retenues"] for l in L if l["retenues"] > 0]
            tot = sum(pond)
            r = reglage(dict(VARIANTES)[nom])
            out.append(
                {
                    "variante": nom,
                    "polarite": pol,
                    "filtre_km": r["filtre_km"],
                    "contour_mm": round(1000 * r["contour_m"], 3),
                    "erreur_forme_pourcent": r["erreur_forme"],
                    "jours": n,
                    "identifiees_par_jour": round(ident / n, 4),
                    "identifiees_dans_le_domaine_par_jour": round(
                        sum(l["identifiees_domaine"] for l in L) / n, 4
                    ),
                    "retenues_par_jour": round(ret / n, 4),
                    "amplitude_moyenne_cm": (
                        round(100 * sum(a * w for a, w in zip(amp, pond)) / tot, 4)
                        if tot
                        else ""
                    ),
                    "rayon_moyen_km": (
                        round(sum(r * w for r, w in zip(ray, pond)) / tot / 1000, 4)
                        if tot
                        else ""
                    ),
                }
            )
    return out


def to_datetime(np_dt64):
    import numpy as np

    ts = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(ts))


def correct_lon(lons_array):
    import numpy as np

    lons = np.array(lons_array, dtype=float)
    lons = np.where(lons > 180, lons - 360, lons)
    return float(np.mean(lons))


def extract_eddies(obs, date_str, eddy_type, avalees=None):
    import numpy as np

    records = []
    if obs is None or len(obs) == 0:
        return records
    for k in range(len(obs)):
        try:
            lat = float(np.mean(obs["contour_lat_s"][k]))
            lon = correct_lon(obs["contour_lon_s"][k])
            if not (-6 < lon < 11) or not (34 < lat < 42):
                continue
            records.append(
                {
                    "date": date_str,
                    "type": eddy_type,
                    "lon": round(lon, 4),
                    "lat": round(lat, 4),
                    "amplitude": float(obs["amplitude"][k]),
                    "radius_s": float(obs["radius_s"][k]),
                }
            )
        except Exception as e:
            if avalees is not None and not avalees:
                avalees.append(f"{type(e).__name__}: {e}")
    return records


def dans_le_domaine(obs):
    import numpy as np

    if obs is None or len(obs) == 0:
        return [], []
    lon = np.where(
        np.asarray(obs["lon"], float) > 180,
        np.asarray(obs["lon"], float) - 360,
        np.asarray(obs["lon"], float),
    )
    lat = np.asarray(obs["lat"], float)
    m = (lon > -6) & (lon < 11) & (lat > 34) & (lat < 42)
    return list(np.asarray(obs["amplitude"], float)[m]), list(
        np.asarray(obs["radius_s"], float)[m]
    )


def lire_atlas():
    atlas = {}
    for nom in ("anticyclones_1993_2023_ALL.csv", "cyclones_1993_2023_ALL.csv"):
        p = os.path.join(ATLAS_REFERENCE, nom)
        if not os.path.isfile(p):
            raise SystemExit(
                f"STOP: regenerated eddy catalogue not found: {p}\n"
                "      run eddy_atlas.py --produire first"
            )
        with io.open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                atlas.setdefault((r["date"], r["type"]), []).append(
                    {
                        "lon": float(r["lon"]),
                        "lat": float(r["lat"]),
                        "amplitude": float(r["amplitude"]),
                    }
                )
    return atlas


def produire(pas):
    import numpy as np
    import xarray as xr
    from py_eddy_tracker.dataset.grid import RegularGridDataset
    from eddy_atlas import matplotlib_compatible, temoin_tient

    compatible, version = matplotlib_compatible()
    if not compatible:
        raise SystemExit(
            f"STOP: matplotlib {version}: py-eddy-tracker 3.6.1 sees "
            "a single contour per level. matplotlib < 3.8 is required."
        )
    temoin_tient()
    racine = os.environ.get("ABASIN_DATA", "")
    chemin = os.path.join(racine, FICHIER_ALLSAT)
    if not os.path.isfile(chemin):
        raise SystemExit(f"STOP: ALL-SAT file not found: {chemin} (ABASIN_DATA)")
    ds = xr.open_dataset(chemin)
    manque = [v for v in ("adt", "ugos", "vgos") if v not in ds.data_vars]
    if manque:
        raise SystemExit(f"STOP: variables not found: {manque}")
    atlas = lire_atlas()
    jours = jours_echantillon(len(ds.time), pas)
    lignes, desaccords = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for n, i in enumerate(jours):
            date = to_datetime(ds.time.values[i])
            date_str = date.strftime("%Y-%m-%d")
            f = os.path.join(tmp, f"adt_{date_str}.nc")
            ds[["adt", "ugos", "vgos"]].isel(time=i).to_netcdf(f)
            for nom, var in VARIANTES:
                r = reglage(var)
                echec = ""
                try:
                    g = RegularGridDataset(f, "longitude", "latitude")
                    g.bessel_high_filter("adt", r["filtre_km"])
                    a, c = g.eddy_identification(
                        "adt", "ugos", "vgos", date, r["contour_m"], r["erreur_forme"]
                    )
                except Exception as e:
                    a = c = None
                    echec = f"{type(e).__name__}: {e}"
                for obs, pol in ((a, "anticyclone"), (c, "cyclone")):
                    avalees = []
                    ret = extract_eddies(obs, date_str, pol, avalees)
                    amp, ray = dans_le_domaine(obs)
                    if nom == "reference":
                        d = comparer_a_l_atlas(ret, atlas.get((date_str, pol), []))
                        if d:
                            desaccords.append(f"{date_str} {pol} : {d}")
                    lignes.append(
                        {
                            "date": date_str,
                            "variante": nom,
                            "polarite": pol,
                            "identifiees": 0 if obs is None else len(obs),
                            "identifiees_domaine": len(amp),
                            "retenues": len(ret),
                            "amplitude_moyenne_m": (
                                float(np.mean([x["amplitude"] for x in ret]))
                                if ret
                                else 0.0
                            ),
                            "rayon_moyen_m": (
                                float(np.mean([x["radius_s"] for x in ret]))
                                if ret
                                else 0.0
                            ),
                            "exception_avalee": avalees[0] if avalees else "",
                            "echec_identification": echec,
                        }
                    )
    ds.close()
    if desaccords:
        return 1
    with io.open(SORTIE_JOURS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    b = bilan(lignes)
    with io.open(SORTIE_BILAN, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b[0]))
        w.writeheader()
        w.writerows(b)
    return 0


def rebatir_bilan():
    with io.open(SORTIE_JOURS, encoding="utf-8", newline="") as f:
        lignes = []
        for r in csv.DictReader(f):
            for k in ("identifiees", "identifiees_domaine", "retenues"):
                r[k] = int(r[k])
            for k in ("amplitude_moyenne_m", "rayon_moyen_m"):
                r[k] = float(r[k])
            lignes.append(r)
    b = bilan(lignes)
    with io.open(SORTIE_BILAN, encoding="utf-8", newline="") as f:
        ancien = list(csv.DictReader(f))
    communs = [k for k in ancien[0] if k not in ("variante", "polarite")]
    for x, y in zip(ancien, b):
        for k in communs:
            if abs(float(x[k]) - float(y[k])) > 1e-9:
                raise SystemExit(
                    f"STOP: {x['variante']} {x['polarite']} {k}: "
                    f"{x[k]} written, {y[k]} rebuilt"
                )
    with io.open(SORTIE_BILAN, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(b[0]))
        w.writeheader()
        w.writerows(b)
    return 0


def diagnostiquer(date_str):
    import numpy as np
    import xarray as xr
    from eddy_atlas import identifier

    chemin = os.path.join(os.environ.get("ABASIN_DATA", ""), FICHIER_ALLSAT)
    ds = xr.open_dataset(chemin)
    i = int(
        np.where(ds.time.values.astype("datetime64[D]").astype(str) == date_str)[0][0]
    )
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "jour.nc")
        ds[["adt", "ugos", "vgos"]].isel(time=i).to_netcdf(f)
        a, c = identifier(f, to_datetime(ds.time.values[i]))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--produire" in a:
        pas = int(a[a.index("--pas") + 1]) if "--pas" in a else PAS_DEFAUT
        sys.exit(produire(pas))
    if "--bilan" in a:
        sys.exit(rebatir_bilan())
    if "--diagnostiquer" in a:
        sys.exit(diagnostiquer(a[a.index("--diagnostiquer") + 1]))
    print(__doc__)
