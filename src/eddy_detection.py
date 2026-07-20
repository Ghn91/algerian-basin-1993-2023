# Mesoscale eddy detection with py-eddy-tracker
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

import xarray as xr
import numpy as np
import pandas as pd
import os, warnings
from datetime import datetime
from tqdm import tqdm
warnings.filterwarnings('ignore')

from py_eddy_tracker.dataset.grid import RegularGridDataset

EDDY_DIR = DATA_DIR + r"\EDDIES"
TEMP_DIR = EDDY_DIR + r"\TEMP"
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Supprimer les anciens CSV vides ───────────────────────────────────────
print("Nettoyage des anciens CSV vides...")
deleted = 0
for yr in range(1993, 2024):
    for prefix in ["anticyclones", "cyclones"]:
        f = os.path.join(EDDY_DIR, f"{prefix}_{yr}.csv")
        if os.path.exists(f) and os.path.getsize(f) < 50:
            os.remove(f)
            deleted += 1
print(f"  {deleted} fichiers vides supprimés\n")

# ── Fonctions utilitaires ─────────────────────────────────────────────────
def to_datetime(np_dt64):
    ts = (np_dt64 - np.datetime64('1970-01-01T00:00:00')) \
         / np.timedelta64(1, 's')
    return datetime.utcfromtimestamp(float(ts))

def correct_lon(lons_array):
    lons = np.array(lons_array, dtype=float)
    lons = np.where(lons > 180, lons - 360, lons)
    return float(np.mean(lons))

def extract_eddies(obs, date_str, eddy_type):
    records = []
    if obs is None or len(obs) == 0:
        return records
    for k in range(len(obs)):
        try:
            lat = float(np.mean(obs['contour_lat_s'][k]))
            lon = correct_lon(obs['contour_lon_s'][k])
            if not (-6 < lon < 11) or not (34 < lat < 42):
                continue
            records.append({
                'date':          date_str,
                'type':          eddy_type,
                'lon':           round(lon, 4),
                'lat':           round(lat, 4),
                'amplitude':     float(obs['amplitude'][k]),
                'radius_s':      float(obs['radius_s'][k]),
                'radius_e':      float(obs['radius_e'][k]),
                'speed_average': float(obs['speed_average'][k]),
                'speed_area':    float(obs['speed_area'][k]),
                'effective_area':float(obs['effective_area'][k]),
                'shape_error_s': float(obs['shape_error_s'][k]),
            })
        except Exception:
            pass
    return records

# ── Chargement ────────────────────────────────────────────────────────────
print("Chargement allsat 1993–2023...")
ds = xr.open_dataset(
    DATA_DIR + r"\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg"
               r"_P1D_multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)
print(f"  {len(ds.time)} pas de temps\n")

# ── Années à traiter (reprise automatique si interruption) ────────────────
years_todo = []
for yr in range(1993, 2024):
    fa = os.path.join(EDDY_DIR, f"anticyclones_{yr}.csv")
    fc = os.path.join(EDDY_DIR, f"cyclones_{yr}.csv")
    if os.path.exists(fa) and os.path.exists(fc) \
       and os.path.getsize(fa) > 50:
        print(f"  {yr} : déjà traité ✓")
    else:
        years_todo.append(yr)

print(f"\nAnnées restantes : {len(years_todo)}/31")
print("Début de la détection...\n")

# ── DÉTECTION COMPLÈTE ────────────────────────────────────────────────────
summary = []

for yr in years_todo:
    print(f"── {yr} ", end='', flush=True)

    times_yr  = ds.time.sel(time=str(yr))
    idx_start = int(np.where(ds.time.values == times_yr.values[0])[0][0])
    n_days    = len(times_yr)

    anti_recs, cyc_recs = [], []
    errors = 0

    for i in tqdm(range(idx_start, idx_start + n_days),
                  desc=f"{yr}", unit="j", leave=False, ncols=70):

        np_date  = ds.time.values[i]
        date     = to_datetime(np_date)
        date_str = date.strftime('%Y-%m-%d')
        tmp_file = os.path.join(TEMP_DIR, f"adt_{date_str}.nc")

        try:
            ds[['adt','ugos','vgos']].isel(time=i).to_netcdf(tmp_file)

            g = RegularGridDataset(tmp_file, 'longitude', 'latitude')
            g.bessel_high_filter('adt', 700)

            a, c = g.eddy_identification(
                'adt', 'ugos', 'vgos', date, 0.002, 55
            )

            anti_recs.extend(extract_eddies(a, date_str, 'anticyclone'))
            cyc_recs.extend( extract_eddies(c, date_str, 'cyclone'))

        except Exception:
            errors += 1
        finally:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

    # ── Sauvegarde CSV annuelle ───────────────────────────────────────────
    pd.DataFrame(anti_recs).to_csv(
        os.path.join(EDDY_DIR, f"anticyclones_{yr}.csv"), index=False
    )
    pd.DataFrame(cyc_recs).to_csv(
        os.path.join(EDDY_DIR, f"cyclones_{yr}.csv"), index=False
    )

    na, nc = len(anti_recs), len(cyc_recs)
    print(f"→ {na:4d} anti | {nc:4d} cyc | err={errors}")
    summary.append({'year':yr,'n_anti':na,'n_cyc':nc,'errors':errors})

# ── Assemblage global ─────────────────────────────────────────────────────
print("\nAssemblage global...")
all_a, all_c = [], []
for yr in range(1993, 2024):
    for prefix, lst in [("anticyclones", all_a), ("cyclones", all_c)]:
        f = os.path.join(EDDY_DIR, f"{prefix}_{yr}.csv")
        if os.path.exists(f) and os.path.getsize(f) > 50:
            try:
                df = pd.read_csv(f)
                if len(df) > 0:
                    lst.append(df)
            except Exception:
                pass

df_a = pd.concat(all_a, ignore_index=True) if all_a else pd.DataFrame()
df_c = pd.concat(all_c, ignore_index=True) if all_c else pd.DataFrame()

df_a.to_csv(os.path.join(EDDY_DIR,"anticyclones_1993_2023_ALL.csv"), index=False)
df_c.to_csv(os.path.join(EDDY_DIR,"cyclones_1993_2023_ALL.csv"),     index=False)

# ── Résumé ────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ATLAS EDDIES — BASSIN ALGÉRIEN 1993–2023")
print("=" * 55)
print(f"  Total anticyclones : {len(df_a):,}")
print(f"  Total cyclones     : {len(df_c):,}")

if len(df_a) > 0:
    print(f"\n  Anticyclones :")
    print(f"    Amplitude  : {df_a['amplitude'].mean()*100:.2f} ± "
          f"{df_a['amplitude'].std()*100:.2f} cm")
    print(f"    Rayon      : {df_a['radius_s'].mean()/1000:.1f} ± "
          f"{df_a['radius_s'].std()/1000:.1f} km")
    print(f"    Vitesse    : {df_a['speed_average'].mean()*100:.2f} ± "
          f"{df_a['speed_average'].std()*100:.2f} cm/s")

if len(df_c) > 0:
    print(f"\n  Cyclones :")
    print(f"    Amplitude  : {df_c['amplitude'].mean()*100:.2f} ± "
          f"{df_c['amplitude'].std()*100:.2f} cm")
    print(f"    Rayon      : {df_c['radius_s'].mean()/1000:.1f} ± "
          f"{df_c['radius_s'].std()/1000:.1f} km")
    print(f"    Vitesse    : {df_c['speed_average'].mean()*100:.2f} ± "
          f"{df_c['speed_average'].std()*100:.2f} cm/s")

print("=" * 55)
ds.close()
