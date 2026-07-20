# Inventory of the local NetCDF files
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

"""
DIAGNOSTIC — Inspecte tous vos fichiers NetCDF CMEMS
Exécuter AVANT animation_READY_TO_RUN.py
Durée : 30 secondes

Usage : python diagnostic_cmems.py
"""
import xarray as xr
import os


fichiers = [f for f in os.listdir(BASE) if f.endswith('.nc')]

print("=" * 70)
print("DIAGNOSTIC FICHIERS CMEMS")
print("=" * 70)

for fname in sorted(fichiers):
    fpath = os.path.join(BASE, fname)
    taille_mo = os.path.getsize(fpath) / 1e6
    print(f"\n{'─'*70}")
    print(f"FICHIER : {fname}")
    print(f"Taille  : {taille_mo:.0f} Mo")
    try:
        ds = xr.open_dataset(fpath)
        print(f"Variables  : {list(ds.data_vars)}")
        print(f"Dimensions : {dict(ds.dims)}")
        print(f"Coordonnées: {list(ds.coords)}")
        if 'time' in ds.dims:
            print(f"Période    : {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")
            print(f"Nb temps   : {len(ds.time)}")
        for var in list(ds.data_vars)[:3]:
            da = ds[var]
            print(f"  [{var}] shape={da.shape}  dtype={da.dtype}  units={da.attrs.get('units','?')}")
        ds.close()
    except Exception as e:
        print(f"ERREUR : {e}")

print("\n" + "=" * 70)
print("CORRESPONDANCE ATTENDUE POUR L'ANIMATION :")
print("=" * 70)
print("""
  c3s_obs-sl_glo..._twosat...nc
    → SLA (article)  Variables attendues : sla, ugos, vgos
    → Résolution 0.25° · Journalier

  cmems_obs-sl_glo..._allsat...nc
    → SLA ALL-SAT    Variables attendues : sla, ugos, vgos
    → Résolution 0.125° · Journalier

  cmems_mod_med_phy-cur..._uo-vo...nc
    → Courants réanalyse  Variables : uo, vo
    → Résolution 4.2km · Dimension depth

  cmems_mod_med_phy-ssh..._zos...nc
    → SSH réanalyse  Variable : zos
    → Résolution 4.2km

  cmems_mod_med_phy-temp..._thetao...nc
    → Température    Variable : thetao
    → Résolution 4.2km · Dimension depth

  cmems_mod_med_bgc-plankton..._chl-phyc...nc
    → Chlorophylle   Variables : chl, phyc
    → Résolution 4.2km · 1999–2023

  cmems_SST_MED_SST_L4...nc
    → SST observée   Variable : analysed_sst (en Kelvin → -273.15)
    → Résolution ~4km
""")
print("Copiez-collez ces informations si l'animation plante sur un KeyError.")
