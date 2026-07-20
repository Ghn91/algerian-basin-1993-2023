# Assemble the eddy catalogue and compute Table 1 statistics
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

import pandas as pd
import numpy as np
import os

EDDY_DIR = DATA_DIR + r"\EDDIES"

# ── Chargement ────────────────────────────────────────────────────────────
print("Chargement des fichiers globaux...")

df_a = pd.read_csv(os.path.join(EDDY_DIR, "anticyclones_1993_2023_ALL.csv"))
df_c = pd.read_csv(os.path.join(EDDY_DIR, "cyclones_1993_2023_ALL.csv"))

df_a['date'] = pd.to_datetime(df_a['date'])
df_c['date'] = pd.to_datetime(df_c['date'])
df_a['year']  = df_a['date'].dt.year
df_c['year']  = df_c['date'].dt.year
df_a['month'] = df_a['date'].dt.month
df_c['month'] = df_c['date'].dt.month

# ── Vérification 1 : couverture temporelle ────────────────────────────────
print("\n" + "="*60)
print("1. COUVERTURE TEMPORELLE")
print("="*60)
print(f"  Anticyclones : {df_a['date'].min().date()} → "
      f"{df_a['date'].max().date()}")
print(f"  Cyclones     : {df_c['date'].min().date()} → "
      f"{df_c['date'].max().date()}")

# Années manquantes ?
years_a = set(df_a['year'].unique())
years_c = set(df_c['year'].unique())
years_all = set(range(1993, 2024))
missing_a = years_all - years_a
missing_c = years_all - years_c
print(f"  Années manquantes anti : "
      f"{sorted(missing_a) if missing_a else 'aucune ✓'}")
print(f"  Années manquantes cyc  : "
      f"{sorted(missing_c) if missing_c else 'aucune ✓'}")

# ── Vérification 2 : coordonnées ──────────────────────────────────────────
print("\n" + "="*60)
print("2. COORDONNÉES — domaine attendu [-5,10]°E · [35,41]°N")
print("="*60)
for label, df in [("Anticyclones", df_a), ("Cyclones", df_c)]:
    lon_ok = ((df['lon'] >= -5.5) & (df['lon'] <= 10.5)).all()
    lat_ok = ((df['lat'] >= 34.5) & (df['lat'] <= 41.5)).all()
    print(f"  {label} :")
    print(f"    Lon : [{df['lon'].min():.2f} → {df['lon'].max():.2f}]"
          f" {'✓' if lon_ok else '❌ HORS DOMAINE'}")
    print(f"    Lat : [{df['lat'].min():.2f} → {df['lat'].max():.2f}]"
          f" {'✓' if lat_ok else '❌ HORS DOMAINE'}")

# ── Vérification 3 : valeurs aberrantes ───────────────────────────────────
print("\n" + "="*60)
print("3. VALEURS ABERRANTES")
print("="*60)
for label, df in [("Anticyclones", df_a), ("Cyclones", df_c)]:
    amp_max  = df['amplitude'].max() * 100      # cm
    rad_max  = df['radius_s'].max()  / 1000     # km
    speed_max= df['speed_average'].max() * 100  # cm/s
    nan_tot  = df.isnull().sum().sum()

    print(f"  {label} :")
    print(f"    Amplitude max  : {amp_max:.1f} cm "
          f"{'✓' if amp_max < 50 else '⚠ vérifier'}")
    print(f"    Rayon max      : {rad_max:.1f} km "
          f"{'✓' if rad_max < 300 else '⚠ vérifier'}")
    print(f"    Vitesse max    : {speed_max:.1f} cm/s "
          f"{'✓' if speed_max < 150 else '⚠ vérifier'}")
    print(f"    Valeurs NaN    : {nan_tot} "
          f"{'✓' if nan_tot == 0 else '⚠ vérifier'}")

# ── Vérification 4 : statistiques globales ────────────────────────────────
print("\n" + "="*60)
print("4. STATISTIQUES GLOBALES")
print("="*60)
for label, df in [("Anticyclones", df_a), ("Cyclones", df_c)]:
    print(f"\n  {label} — {len(df):,} détections totales")
    print(f"    Amplitude  : {df['amplitude'].mean()*100:.2f} ± "
          f"{df['amplitude'].std()*100:.2f} cm")
    print(f"    Rayon (s)  : {df['radius_s'].mean()/1000:.1f} ± "
          f"{df['radius_s'].std()/1000:.1f} km")
    print(f"    Rayon (e)  : {df['radius_e'].mean()/1000:.1f} ± "
          f"{df['radius_e'].std()/1000:.1f} km")
    print(f"    Vitesse    : {df['speed_average'].mean()*100:.2f} ± "
          f"{df['speed_average'].std()*100:.2f} cm/s")
    print(f"    Moy/jour   : {len(df)/11322:.2f} détections/jour")
    print(f"    Moy/an     : {len(df)/31:.0f} détections/an")

# ── Vérification 5 : distribution annuelle ───────────────────────────────
print("\n" + "="*60)
print("5. DISTRIBUTION ANNUELLE")
print("="*60)
print(f"  {'Année':<8} {'Anti':>8} {'Cyc':>8} {'Total':>8}")
print(f"  {'-'*32}")
for yr in range(1993, 2024):
    na = len(df_a[df_a['year'] == yr])
    nc = len(df_c[df_c['year'] == yr])
    flag = " ⚠" if na == 0 or nc == 0 else ""
    print(f"  {yr:<8} {na:>8} {nc:>8} {na+nc:>8}{flag}")

# ── Vérification 6 : cycle saisonnier ────────────────────────────────────
print("\n" + "="*60)
print("6. CYCLE SAISONNIER")
print("="*60)
month_names = ['Jan','Fév','Mar','Avr','Mai','Jun',
               'Jul','Aoû','Sep','Oct','Nov','Déc']
for label, df in [("Anticyclones", df_a), ("Cyclones", df_c)]:
    by_month = df.groupby('month').size()
    max_m = int(by_month.idxmax())
    min_m = int(by_month.idxmin())
    print(f"  {label} : max={month_names[max_m-1]} "
          f"({by_month.max()}) | min={month_names[min_m-1]} "
          f"({by_month.min()})")

print("\n" + "="*60)
print("VÉRIFICATION TERMINÉE")
print("="*60)

# ----------------------------------------------------------------------
import pandas as pd
import numpy as np

EDDY_DIR = r"C:\Users\kikk\CLAUDE ARTICLE DATA\EDDIES"

df_a = pd.read_csv(EDDY_DIR + r"\anticyclones_1993_2023_ALL.csv")
df_c = pd.read_csv(EDDY_DIR + r"\cyclones_1993_2023_ALL.csv")

for df in [df_a, df_c]:
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year

def stats(df, name):
    print(f"\n=== {name} ===")
    print(f"Total détections       : {len(df):,}")

    # Amplitude
    amp = df['amplitude'] * 100        # m → cm
    print(f"Amplitude moyenne (cm) : {amp.mean():.2f} ± {amp.std():.2f}")

    # Rayon de vitesse max (radius_s)
    r = df['radius_s'] / 1000          # m → km
    print(f"Rayon speed (km)       : {r.mean():.1f} ± {r.std():.1f}")

    # Vitesse azimutale max
    # py-eddy-tracker stocke 'speed_average' ou 'speed_radius' selon la version
    # essayons les deux noms possibles
    for col in ['speed_average', 'speed_radius', 'uavg']:
        if col in df.columns:
            spd = df[col] * 100        # m/s → cm/s
            print(f"Vitesse max ({col}) cm/s: {spd.mean():.2f} ± {spd.std():.2f}")
            break

    # Comptes annuels
    annual = df.groupby('year').size()
    min_yr = annual.idxmin()
    max_yr = annual.idxmax()
    print(f"Min annuel             : {annual[min_yr]} ({min_yr})")
    print(f"Max annuel             : {annual[max_yr]} ({max_yr})")

    # Toutes les colonnes disponibles
    print(f"Colonnes disponibles   : {list(df.columns)}")

stats(df_a, "ANTICYCLONES")
stats(df_c, "CYCLONES")
