"""
WeMO (published UB Barcelona) and NAO cross-correlation with EKE.

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
import pandas as pd
from scipy import stats
import urllib.request


# EKE TWO-SAT
print("Calcul EKE...")
ds2 = xr.open_dataset(
    DATA_DIR + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg"
               r"_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc"
)
ugos_mean = ds2['ugos'].mean('time')
vgos_mean = ds2['vgos'].mean('time')
eke_monthly = (0.5*((ds2['ugos']-ugos_mean)**2 +
                    (ds2['vgos']-vgos_mean)**2)*1e4
               ).mean(dim=['latitude','longitude'], skipna=True
               ).resample(time='ME').mean()
eke_clim = eke_monthly.groupby('time.month').mean()
eke_anom = eke_monthly.groupby('time.month') - eke_clim
ds2.close()
eke_vals  = eke_anom.values
eke_dates = pd.DatetimeIndex(eke_anom.time.values).to_period('M')
print(f"EKE : {len(eke_vals)} mois")

# WeMO 1993-2013 (données UB intégrées directement)
wemo_data = {
    1993:[-0.19,-1.18,-0.99, 0.65,-0.72, 1.23, 0.11,-0.38, 1.00,-0.47,-0.84, 2.06],
    1994:[ 0.91, 0.31, 0.37, 3.19, 0.67,-0.04,-0.41, 0.75, 0.07,-0.96, 0.06, 0.45],
    1995:[ 1.48, 1.14, 0.37,-0.39, 0.63,-0.53,-0.59,-0.10, 1.91,-0.71,-0.59,-0.63],
    1996:[-2.25, 0.61,-1.72,-0.47, 0.63,-0.38, 0.56, 1.45, 1.11, 1.35, 1.71,-1.43],
    1997:[-1.88, 0.55,-0.40,-1.09,-0.11, 0.68, 1.27, 0.37,-1.58,-0.11, 0.07, 0.17],
    1998:[-0.51,-1.21,-0.13, 1.95,-1.37,-0.15, 0.99,-0.45, 0.06, 2.51, 0.59,-0.85],
    1999:[-0.49, 1.08,-0.55, 1.20, 0.33, 0.68, 0.17, 0.02, 0.25, 0.04,-0.26, 1.52],
    2000:[-0.65, 0.70, 0.30, 0.10,-0.10, 0.20, 1.99, 0.30, 0.68, 1.00, 1.45,-0.21],
    2001:[ 0.10, 0.32, 0.91, 2.42, 0.15, 0.84, 1.22, 0.72, 0.49,-0.38,-0.35,-1.03],
    2002:[-0.70, 0.84,-0.80,-0.41, 0.45, 0.26, 0.70, 1.31, 0.10, 1.13, 1.18, 0.11],
    2003:[ 1.22,-0.76,-1.72,-1.82,-0.03,-0.77, 1.60, 0.91, 0.82,-0.17,-1.27,-0.39],
    2004:[ 1.17,-0.65,-0.77,-0.75,-0.49,-0.31,-0.40, 0.97, 0.08,-0.17, 0.04,-0.34],
    2005:[ 0.31,-0.04,-1.07,-0.70,-0.85,-2.27, 0.10,-0.84,-0.38,-2.18,-1.25,-0.08],
    2006:[-1.55,-0.02, 0.86,-1.41,-1.49,-3.18,-2.03, 1.55,-0.39,-1.03,-0.61,-0.66],
    2007:[ 0.44, 0.72, 1.08,-2.95, 1.48, 0.01, 0.30,-0.96,-0.63,-1.22,-0.62,-0.56],
    2008:[-0.30,-2.09, 1.81, 0.03,-2.42,-1.06, 0.68,-0.06,-1.63,-0.51, 0.12,-0.17],
    2009:[ 0.23,-0.29,-0.51, 0.63,-1.00, 0.87, 1.11,-1.59,-1.26,-0.36, 0.92, 0.12],
    2010:[-0.19,-0.70,-0.96,-2.40, 0.97,-0.07,-1.65,-0.36,-0.46,-0.91, 1.24,-0.31],
    2011:[-0.91, 0.72,-2.14,-3.08,-1.48,-0.88,-0.03,-0.97,-0.28,-0.95,-2.62, 1.58],
    2012:[ 0.48, 0.38,-0.42, 1.28,-0.41,-0.03,-0.16,-0.58,-1.53,-0.85,-0.98, 1.06],
    2013:[ 1.58, 1.06,-0.81,-0.98, 1.90,-0.37,-2.27,-1.93, 0.03,-0.64, 1.17,-0.49],
}

wemo_rows = []
for yr, vals in wemo_data.items():
    for m, v in enumerate(vals, 1):
        wemo_rows.append(pd.Period(year=yr, month=m, freq='M'))
wemo_index = pd.PeriodIndex(wemo_rows)
wemo_vals  = np.array([v for vals in wemo_data.values() for v in vals])
wemo_vals  = (wemo_vals - wemo_vals.mean()) / wemo_vals.std()  # standardiser
print(f"WeMO : {len(wemo_vals)} mois (1993-2013, source UB Barcelona)")

# NAO mensuel NOAA (automatique, 1993-2023)
print("Téléchargement NAO NOAA...")
try:
    nao_url = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/"
               "pna/norm.nao.monthly.b5001.current.ascii.table")
    with urllib.request.urlopen(nao_url, timeout=20) as r:
        nao_text = r.read().decode('utf-8')

    nao_rows, nao_dates = [], []
    for line in nao_text.strip().split('\n')[1:]:
        parts = line.split()
        if len(parts) < 13: continue
        yr = int(float(parts[0]))
        if yr < 1993 or yr > 2023: continue
        for m in range(1, 13):
            nao_dates.append(pd.Period(year=yr, month=m, freq='M'))
            nao_rows.append(float(parts[m]))
    nao_index = pd.PeriodIndex(nao_dates)
    nao_vals  = np.array(nao_rows)
    print(f"NAO  : {len(nao_vals)} mois (1993-2023, source NOAA CPC)")
    nao_ok = True
except Exception as e:
    print(f"NAO non disponible : {e}")
    nao_ok = False

# Corrélation croisée
def cross_corr(eke_d, eke_v, idx_d, idx_v, name, max_lag=24):
    rows = []
    for lag in range(-max_lag, max_lag+1):
        # décaler l'indice de `lag` mois
        idx_d_shifted = idx_d.shift(lag)
        common = np.intersect1d(eke_d, idx_d_shifted)
        if len(common) < 30: continue
        eke_mask = np.isin(eke_d, common)
        idx_mask = np.isin(idx_d_shifted, common)
        x = eke_v[eke_mask]
        y = idx_v[idx_mask]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 30: continue
        r, p = stats.pearsonr(x[mask], y[mask])
        rows.append({'lag': lag, 'r': r, 'p': p, 'n': mask.sum()})

    df  = pd.DataFrame(rows)
    pk  = df.loc[df['r'].abs().idxmax()]
    r0r = df.loc[df['lag']==0]
    r0, p0, n0 = r0r['r'].values[0], r0r['p'].values[0], r0r['n'].values[0]
    sig = df[(df['p'] < 0.05) & df['lag'].between(-12, 12)]

    print(f"\n{'='*52}")
    print(f"  {name}")
    print(f"  n (lag 0)  = {n0} mois")
    print(f"  r (lag 0)  = {r0:+.3f}   p = {p0:.4f}")
    print(f"  r max      = {pk['r']:+.3f}   p = {pk['p']:.4f}"
          f"   lag = {int(pk['lag'])} mois")
    if not sig.empty:
        print(f"  Lags sign. (p<0.05, |lag|≤12) : "
              f"{int(sig['lag'].min())} à {int(sig['lag'].max())} mois")
    else:
        print(f"  Aucun lag significatif à p<0.05 dans ±12 mois")
    return df, pk, r0, p0

df_w, pk_w, r0_w, p0_w = cross_corr(
    eke_dates, eke_vals, wemo_index, wemo_vals,
    "WeMO vs EKE  (UB 1993-2013, n=252)")

if nao_ok:
    df_n, pk_n, r0_n, p0_n = cross_corr(
        eke_dates, eke_vals, nao_index, nao_vals,
        "NAO  vs EKE  (NOAA 1993-2023, n=372)")

# Sortie finale
print(f"""
{'='*60}
RÉSULTATS POUR main.tex — section 5.3
{'='*60}

WeMO : lag 0  r = {r0_w:+.3f}  p = {p0_w:.4f}
       peak   r = {pk_w['r']:+.3f}  p = {pk_w['p']:.4f}  lag = {int(pk_w['lag'])} mois
""")
if nao_ok:
    print(f"NAO  : lag 0  r = {r0_n:+.3f}  p = {p0_n:.4f}")
    print(f"       peak   r = {pk_n['r']:+.3f}  p = {pk_n['p']:.4f}  lag = {int(pk_n['lag'])} mois")
