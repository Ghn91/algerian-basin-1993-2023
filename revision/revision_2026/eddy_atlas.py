"""Daily eddy catalogue of the study domain, 1993-2023: py-eddy-tracker 3.6.1
run on every daily ADT map of the ALL-SAT product (filter cut-off 700 km,
contour interval 2 mm, shape error 55 %), detections outside the domain
excluded. Refuses to run with matplotlib >= 3.8, with which py-eddy-tracker
3.6.1 sees a single contour per level, and runs a synthetic field with known
eddies before producing. Writes the catalogue to sorties/eddy_atlas/.
Requires the environment of environment_detection.yml.

Usage: python eddy_atlas.py --produire | --comparer DIR | --soumis
"""

import csv
import io
import os
import sys
import tempfile
from datetime import datetime

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SORTIES = os.path.join(DOSSIER, "sorties", "eddy_atlas")
FICHIER_ALLSAT = (
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_"
    "multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)
COLONNES = [
    "date",
    "type",
    "lon",
    "lat",
    "amplitude",
    "radius_s",
    "radius_e",
    "speed_average",
    "speed_area",
    "effective_area",
    "shape_error_s",
]
FILTRE_KM, CONTOUR_M, ERREUR_FORME = 700, 0.002, 55

ATLAS_FICHIERS = {
    "anticyclone": os.path.join(SORTIES, "anticyclones_1993_2023_ALL.csv"),
    "cyclone": os.path.join(SORTIES, "cyclones_1993_2023_ALL.csv"),
}


def matplotlib_compatible():
    import matplotlib

    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return (major, minor) < (3, 8), matplotlib.__version__


def to_datetime(np_dt64):
    import numpy as np

    ts = (np_dt64 - np.datetime64("1970-01-01T00:00:00")) / np.timedelta64(1, "s")
    return datetime.utcfromtimestamp(float(ts))


def correct_lon(lons_array):
    import numpy as np

    lons = np.array(lons_array, dtype=float)
    lons = np.where(lons > 180, lons - 360, lons)
    return float(np.mean(lons))


def extract_eddies(obs, date_str, eddy_type):
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
                    "radius_e": float(obs["radius_e"][k]),
                    "speed_average": float(obs["speed_average"][k]),
                    "speed_area": float(obs["speed_area"][k]),
                    "effective_area": float(obs["effective_area"][k]),
                    "shape_error_s": float(obs["shape_error_s"][k]),
                }
            )
        except Exception:
            pass
    return records


def identifier(chemin_jour, date, filtre=True):
    from py_eddy_tracker.dataset.grid import RegularGridDataset

    g = RegularGridDataset(chemin_jour, "longitude", "latitude")
    if filtre:
        g.bessel_high_filter("adt", FILTRE_KM)
    return g.eddy_identification("adt", "ugos", "vgos", date, CONTOUR_M, ERREUR_FORME)


def produire(pas):
    import xarray as xr

    ok, version = matplotlib_compatible()
    if not ok:
        raise SystemExit(
            f"STOP: matplotlib {version}: py-eddy-tracker 3.6.1 would see a "
            "single contour per level. matplotlib < 3.8 is required."
        )
    temoin_tient()
    chemin = os.path.join(os.environ.get("ABASIN_DATA", ""), FICHIER_ALLSAT)
    if not os.path.isfile(chemin):
        raise SystemExit(f"STOP: ALL-SAT file not found: {chemin} (ABASIN_DATA)")
    ds = xr.open_dataset(chemin)
    manque = [v for v in ("adt", "ugos", "vgos") if v not in ds.data_vars]
    if manque:
        raise SystemExit(f"STOP: variables not found: {manque}")
    os.makedirs(SORTIES, exist_ok=True)
    jours = list(range(0, len(ds.time), pas))
    recs = {"anticyclone": [], "cyclone": []}
    par_jour, echecs = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for n, i in enumerate(jours):
            date = to_datetime(ds.time.values[i])
            date_str = date.strftime("%Y-%m-%d")
            f = os.path.join(tmp, "jour.nc")
            ds[["adt", "ugos", "vgos"]].isel(time=i).to_netcdf(f)
            try:
                a, c = identifier(f, date)
            except Exception:
                a = c = None
                echecs += 1
            ra = extract_eddies(a, date_str, "anticyclone")
            rc = extract_eddies(c, date_str, "cyclone")
            recs["anticyclone"] += ra
            recs["cyclone"] += rc
            par_jour.append(
                {
                    "date": date_str,
                    "identifies_anticyclones": 0 if a is None else len(a),
                    "identifies_cyclones": 0 if c is None else len(c),
                    "retenus_anticyclones": len(ra),
                    "retenus_cyclones": len(rc),
                }
            )
    ds.close()
    suffixe = "" if pas == 1 else f"_un_jour_sur_{pas}"
    for pol, nom in (("anticyclone", "anticyclones"), ("cyclone", "cyclones")):
        p = os.path.join(SORTIES, f"{nom}_1993_2023_ALL{suffixe}.csv")
        with io.open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLONNES)
            w.writeheader()
            w.writerows(recs[pol])
    p = os.path.join(SORTIES, f"comptes_par_jour{suffixe}.csv")
    with io.open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(par_jour[0]))
        w.writeheader()
        w.writerows(par_jour)
    return 0 if echecs == 0 else 1


ATLAS_SOUMIS = os.path.join(os.path.dirname(DOSSIER), "origine_2026-03")
SORTIE_SOUMIS = os.path.join(DOSSIER, "sorties", "eddy_atlas_submitted_counts.csv")


def decompter(lignes):
    from collections import Counter

    c = Counter(r["date"] for r in lignes)
    return len(lignes), len(c), (max(c.values()) if c else 0)


def decompter_soumis():
    out = []
    for pol, nom in (
        ("anticyclone", "anticyclones_1993_2023_ALL.csv"),
        ("cyclone", "cyclones_1993_2023_ALL.csv"),
    ):
        with io.open(
            os.path.join(ATLAS_SOUMIS, nom), encoding="utf-8", newline=""
        ) as f:
            n, d, m = decompter(list(csv.DictReader(f)))
        out.append(
            {
                "polarite": pol,
                "detections": n,
                "dates_distinctes": d,
                "maximum_par_jour": m,
            }
        )
    with io.open(SORTIE_SOUMIS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return 0


CHAMPS_EMPLOYES = ("lon", "lat", "amplitude", "radius_s", "radius_e", "speed_average")


def comparer_atlas(a, b, tol=1e-9):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def par_jour(L):
        d = {}
        for r in L:
            d.setdefault((r["date"], r["type"]), []).append(r)
        return d

    A, B = par_jour(a), par_jour(b)
    out = {
        "jours_compte_different": [],
        "desaccords": {c: 0 for c in CHAMPS_EMPLOYES},
        "ecart_max": {c: 0.0 for c in CHAMPS_EMPLOYES},
        "structures_comparees": 0,
        "deplacements": [],
    }
    for cle in sorted(set(A) | set(B)):
        la, lb = A.get(cle, []), B.get(cle, [])
        if len(la) != len(lb):
            out["jours_compte_different"].append((cle, len(la), len(lb)))
            continue
        pa = np.array([[float(r["lon"]), float(r["lat"])] for r in la]).reshape(-1, 2)
        pb = np.array([[float(r["lon"]), float(r["lat"])] for r in lb]).reshape(-1, 2)
        dist = np.hypot(
            pa[:, None, 0] - pb[None, :, 0], pa[:, None, 1] - pb[None, :, 1]
        )
        ia, ib = linear_sum_assignment(dist)
        out["deplacements"].extend(float(x) for x in dist[ia, ib])
        for ra, rb in ((la[i], lb[j]) for i, j in zip(ia, ib)):
            out["structures_comparees"] += 1
            for c in CHAMPS_EMPLOYES:
                e = abs(float(ra[c]) - float(rb[c]))
                out["ecart_max"][c] = max(out["ecart_max"][c], e)
                if e > tol * max(1.0, abs(float(ra[c]))):
                    out["desaccords"][c] += 1
    return out


def lire_atlas_dossier(dossier):
    L = []
    for nom in ("anticyclones_1993_2023_ALL.csv", "cyclones_1993_2023_ALL.csv"):
        with io.open(os.path.join(dossier, nom), encoding="utf-8", newline="") as f:
            L += list(csv.DictReader(f))
    return L


SORTIE_COMPARAISON = os.path.join(DOSSIER, "sorties", "eddy_atlas_svml_comparison.csv")


def comparer(dossier_autre):
    import numpy as np

    r = comparer_atlas(lire_atlas_dossier(SORTIES), lire_atlas_dossier(dossier_autre))
    dep = np.array(r["deplacements"])
    lignes = [
        {"grandeur": "structures comparees", "valeur": r["structures_comparees"]},
        {
            "grandeur": "jours dont le compte differe",
            "valeur": len(r["jours_compte_different"]),
        },
    ]
    for c in CHAMPS_EMPLOYES:
        lignes.append({"grandeur": f"{c} en desaccord", "valeur": r["desaccords"][c]})
        lignes.append(
            {"grandeur": f"{c} ecart maximal", "valeur": f"{r['ecart_max'][c]:.6f}"}
        )
    if dep.size:
        lignes.append(
            {"grandeur": "deplacement median (deg)", "valeur": f"{np.median(dep):.6f}"}
        )
        lignes.append(
            {"grandeur": "deplacement maximal (deg)", "valeur": f"{dep.max():.6f}"}
        )
    lignes.append(
        {
            "grandeur": "atlas compare",
            "valeur": os.path.basename(os.path.normpath(dossier_autre)),
        }
    )
    with io.open(SORTIE_COMPARAISON, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["grandeur", "valeur"])
        w.writeheader()
        w.writerows(lignes)
    return (
        0
        if not r["jours_compte_different"] and not any(r["desaccords"].values())
        else 1
    )


def champ_temoin(chemin):
    import numpy as np
    import xarray as xr

    lon = np.arange(-5, 10, 0.125)
    lat = np.arange(35, 41, 0.125)
    LO, LA = np.meshgrid(lon, lat)
    bosses = [
        (+0.10, -2.0, 38.0),
        (+0.10, 3.0, 39.0),
        (+0.10, 7.5, 38.0),
        (-0.10, 0.5, 37.5),
        (-0.10, 5.0, 37.5),
    ]
    adt = sum(a * np.exp(-((LO - x) ** 2 + (LA - y) ** 2) / 0.25) for a, x, y in bosses)
    g, om = 9.81, 7.292e-5
    f = 2 * om * np.sin(np.deg2rad(LA))
    dy = 111e3 * 0.125
    dx = dy * np.cos(np.deg2rad(LA))
    ugos = -g / f * np.gradient(adt, axis=0) / dy
    vgos = g / f * np.gradient(adt, axis=1) / dx
    terre = LA < 35.5
    champs = {
        k: np.where(terre, np.nan, v)
        for k, v in dict(adt=adt, ugos=ugos, vgos=vgos).items()
    }
    ds = xr.Dataset(
        {k: (("latitude", "longitude"), v) for k, v in champs.items()},
        coords=dict(latitude=lat, longitude=lon),
    )
    ds.adt.attrs["units"] = "m"
    ds.ugos.attrs["units"] = "m/s"
    ds.vgos.attrs["units"] = "m/s"
    ds.to_netcdf(chemin)
    return 3, 2


def environnement():
    import importlib

    vers = []
    for nom in (
        "matplotlib",
        "numpy",
        "numba",
        "llvmlite",
        "scipy",
        "netCDF4",
        "xarray",
        "py_eddy_tracker",
    ):
        try:
            vers.append(f"{nom} {importlib.import_module(nom).__version__}")
        except Exception as e:
            vers.append(f"{nom} ? ({type(e).__name__})")
    try:
        from numba import config as nconf

        vers.append(f"SVML {'on' if getattr(nconf, 'USING_SVML', False) else 'off'}")
    except Exception:
        pass
    vers.append(
        f"NUMBA_DISABLE_INTEL_SVML={os.environ.get('NUMBA_DISABLE_INTEL_SVML', '')}"
    )
    return " | ".join(vers)


def temoin():
    import math
    import numpy as np

    compatible, version = matplotlib_compatible()
    verifs = [(f"matplotlib older than 3.8 (found {version})", compatible)]
    import importlib.util

    if importlib.util.find_spec("py_eddy_tracker") is None:
        return verifs + [("py-eddy-tracker is installed", False)], []
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "temoin.nc")
        na, nc = champ_temoin(f)
        a, c = identifier(f, datetime(2000, 1, 1), filtre=False)
        ra = extract_eddies(a, "2000-01-01", "anticyclone")
        rc = extract_eddies(c, "2000-01-01", "cyclone")
    toutes = [(o, k) for o in (a, c) for k in range(len(o))]

    def attendu(lat):
        return 39.3 * math.sqrt(math.cos(math.radians(lat)))

    def lat_contour(o, k):
        return float(np.mean(o["contour_lat_s"][k]))

    verifs += [
        (
            f"the synthetic field returns its {na} anticyclones (found: {len(ra)})",
            len(ra) == na,
        ),
        (
            f"the synthetic field returns its {nc} cyclones (found: {len(rc)})",
            len(rc) == nc,
        ),
        (
            "positions match the synthetic eddies within 0.3 deg",
            sorted(round(r["lon"]) for r in ra) == [-2, 3, 8]
            and all(
                abs(r["lat"] - y) < 0.3
                for r, y in zip(sorted(ra, key=lambda r: r["lon"]), (38.0, 39.0, 38.0))
            ),
        ),
        (
            "the stored centre (lat) is that of the contour, within 0.1 deg",
            all(abs(float(o["lat"][k]) - lat_contour(o, k)) < 0.1 for o, k in toutes),
        ),
        (
            "the stored centre (lon, modulo 360) is that of the contour",
            all(
                abs(
                    (
                        (float(o["lon"][k]) - correct_lon(o["contour_lon_s"][k]) + 180)
                        % 360
                    )
                    - 180
                )
                < 0.1
                for o, k in toutes
            ),
        ),
        (
            "radius_s matches the expected radius within 15 % (about 35 km)",
            all(
                0.85
                < float(o["radius_s"][k]) / 1000 / attendu(lat_contour(o, k))
                < 1.15
                for o, k in toutes
            ),
        ),
        (
            "speed_area matches the expected ellipse area within 25 %",
            all(
                0.75
                < float(o["speed_area"][k])
                / (math.pi * (1000 * attendu(lat_contour(o, k))) ** 2)
                < 1.25
                for o, k in toutes
            ),
        ),
        (
            "the amplitude is that of the synthetic eddies, 8 to 10 cm",
            all(0.08 < abs(float(o["amplitude"][k])) <= 0.10 for o, k in toutes),
        ),
        (
            "speed_average lies between the expected meridional and zonal speeds (0.15-0.23 m/s)",
            all(0.15 < float(o["speed_average"][k]) < 0.23 for o, k in toutes),
        ),
        (
            "radius_e is never smaller than radius_s",
            all(float(o["radius_e"][k]) >= float(o["radius_s"][k]) for o, k in toutes),
        ),
    ]
    mesures = [
        (
            float(o["lon"][k]),
            float(o["lat"][k]),
            correct_lon(o["contour_lon_s"][k]),
            lat_contour(o, k),
            float(o["radius_s"][k]) / 1000,
            float(o["speed_area"][k]) / (math.pi * float(o["radius_s"][k]) ** 2),
            float(o["amplitude"][k]),
            float(o["speed_average"][k]),
            float(o["radius_e"][k]) / 1000,
        )
        for o, k in toutes
    ]
    return verifs, mesures


def temoin_tient():
    verifs, _ = temoin()
    defauts = [t for t, b in verifs if not b]
    if defauts:
        raise SystemExit(
            "STOP: the synthetic field does not return its expected values "
            "in this environment; nothing is produced.\n  "
            + "\n  ".join(defauts)
            + "\n  "
            + environnement()
        )
    return True


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--produire" in a:
        pas = int(a[a.index("--pas") + 1]) if "--pas" in a else 1
        sys.exit(produire(pas))
    if "--soumis" in a:
        sys.exit(decompter_soumis())
    if "--comparer" in a:
        sys.exit(comparer(a[a.index("--comparer") + 1]))
    print(__doc__)
