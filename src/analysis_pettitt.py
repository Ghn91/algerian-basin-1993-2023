"""
Pettitt change-point test on the annual TWO-SAT EKE anomaly.

Part of the reproducibility archive for:
Nedjar, Nacef, Houma & Bachari (in prep.), "Thirty years of surface
circulation variability in the Algerian Basin (1993-2023)",
Regional Studies in Marine Science.

Data root and derived paths come from config.py (edit DATA_DIR there,
or set the ABASIN_DATA environment variable).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, BASE  # noqa: F401

import xarray as xr
import numpy as np
from pyhomogeneity import pettitt_test


# Chargement TWO-SAT
print("Chargement TWO-SAT...")
ds2 = xr.open_dataset(
    DATA_DIR + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg"
               r"_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc"
)

# EKE TWO-SAT mensuelle
ugos_mean = ds2['ugos'].mean('time')
vgos_mean = ds2['vgos'].mean('time')
ug_anom   = ds2['ugos'] - ugos_mean
vg_anom   = ds2['vgos'] - vgos_mean

eke_monthly = (0.5*(ug_anom**2 + vg_anom**2)*1e4
               ).mean(dim=['latitude','longitude'], skipna=True)
ds2.close()

# Désaisonnalisation -> anomalie mensuelle
eke_clim  = eke_monthly.groupby('time.month').mean('time')
eke_anom  = eke_monthly.groupby('time.month') - eke_clim

# Anomalie ANNUELLE (moyenne des 12 mois de chaque année)
eke_annual_anom = eke_anom.resample(time='YE').mean()

years  = eke_annual_anom.time.dt.year.values
values = eke_annual_anom.values

print(f"Série annuelle : {len(values)} valeurs ({years[0]}–{years[-1]})")
print(f"r₁ lag-1 : {np.corrcoef(values[:-1], values[1:])[0,1]:.4f}")

# Test de Pettitt
result = pettitt_test(values)

print("\n=== RÉSULTAT TEST DE PETTITT ===")
print(result)
print(f"\nChange point index : {result.cp}")
print(f"Change point year  : {years[result.cp]}")
print(f"U statistic        : {result.U:.1f}")
print(f"p-value            : {result.p:.4f}")
print(f"Significant (p<0.05) : {result.p < 0.05}")

# Phrase à insérer dans main.tex
cp_year = years[result.cp]
print(f"""
=== PHRASE POUR main.tex (section 3.5.1) ===

The Pettitt test {'identifies a statistically significant' if result.p < 0.05 else 'does not identify a statistically significant'} change point; the
candidate year is {cp_year} (U = {result.U:.1f}, p = {result.p:.4f}).
""")
