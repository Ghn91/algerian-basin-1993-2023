"""
Figure 8 - ERA5 WeMO reconstruction and WeMO/EKE cross-correlation.

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

"""
EXTENSION WeMO 1993–2023 DEPUIS ERA5
=====================================
Calcule l'indice WeMO mensuel par reconstruction ERA5
(pression de surface stations San Fernando + Padua),
puis recalcule la corrélation croisée WeMO–EKE sur
la période complète 1993–2023 (n=372 mois).

DÉFINITION WEMO (Martin-Vide & Lopez-Bustins 2006)
  WeMO = P*(Padua) – P*(San Fernando)
  où P* = anomalie de pression normalisée à chaque station
  -> pression réduite au niveau de la mer (MSLP)

STATIONS
  San Fernando : 36.45°N, 6.20°W  (Cadiz, Espagne)
  Padua        : 45.37°N, 11.87°E (Vénétie, Italie)

ÉTAPE 1 — Téléchargement ERA5 (si non déjà fait)
ÉTAPE 2 — Calcul indice WeMO mensuel 1993–2023
ÉTAPE 3 — Corrélation croisée avec EKE TWO-SAT
ÉTAPE 4 — Figure publication-ready

DÉPENDANCES
  pip install cdsapi xarray scipy matplotlib cmocean
  compte CDS (gratuit) : https://cds.climate.copernicus.eu
"""

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

# CHEMINS
ERA5_F  = BASE + r"\ERA5_MSLP_1993_2023_monthly.nc"
EKT_F   = BASE + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc"
OUT_DIR = BASE + r"\FIGURES"
OUT_FIG = OUT_DIR + r"\Figure_WeMO_EKE_1993_2023.pdf"
OUT_CSV = BASE + r"\wemo_index_era5_1993_2023.csv"

# ÉTAPE 1 — TÉLÉCHARGEMENT ERA5 (exécuter une seule fois)
def download_era5_mslp():
    """
    Télécharge ERA5 MSLP mensuel 1993–2023 via CDS API.
    Fichier ~50 Mo. Requiert un compte CDS gratuit.

    Create the file ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api/v2
        key: VOTRE_UID:VOTRE_API_KEY
    Obtenir uid+key sur : https://cds.climate.copernicus.eu/user/register
    """
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'mean_sea_level_pressure',
            'year': [str(y) for y in range(1993, 2024)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
            'area': [50, -10, 30, 20],   # N W S E
            'format': 'netcdf'
        },
        ERA5_F
    )
    print(f"ERA5 MSLP téléchargé : {ERA5_F}")


# ÉTAPE 2 — CALCUL INDICE WeMO MENSUEL
def compute_wemo_index(era5_file):
    """
    Calcule l indice WeMO mensuel depuis ERA5.
    Robust : inspecte automatiquement les dims et coords.
    """
    print("Calcul indice WeMO depuis ERA5...")
    ds = xr.open_dataset(era5_file)

    # Inspection complète du fichier
    print(f"  Variables   : {list(ds.data_vars)}")
    print(f"  Dimensions  : {dict(ds.dims)}")
    print(f"  Coordonnees : {list(ds.coords)}")

    # Nom de la variable MSL
    msl_var = None
    for candidate in ['msl', 'mean_sea_level_pressure', 'sp', 'slp', 'psl']:
        if candidate in ds.data_vars:
            msl_var = candidate
            break
    if msl_var is None:
        msl_var = list(ds.data_vars)[0]
    print(f"  Variable MSL choisie : {msl_var}")

    msl = ds[msl_var]
    print(f"  Shape : {msl.shape}  dims : {msl.dims}")

    # Noms des dimensions
    lon_name = None
    lat_name = None
    time_name = None

    for d in msl.dims:
        dl = d.lower()
        if dl in ('longitude', 'lon', 'x'):
            lon_name = d
        elif dl in ('latitude', 'lat', 'y'):
            lat_name = d
        elif dl in ('time', 'valid_time', 't'):
            time_name = d

    # Fallback : chercher dans les coords du dataset
    if time_name is None:
        for c in ds.coords:
            if c.lower() in ('time', 'valid_time', 't'):
                time_name = c
                break
    if lon_name is None:
        for c in ds.coords:
            if c.lower() in ('longitude', 'lon', 'x'):
                lon_name = c
                break
    if lat_name is None:
        for c in ds.coords:
            if c.lower() in ('latitude', 'lat', 'y'):
                lat_name = c
                break

    print(f"  lon={lon_name}  lat={lat_name}  time={time_name}")

    if time_name is None:
        raise ValueError("Dimension time introuvable. "
                         "Dims disponibles : " + str(list(msl.dims)))

    # Extraction time
    time = msl[time_name].values
    if len(time) == 0:
        raise ValueError("La dimension time est vide — "
                         "le fichier ERA5 est peut-etre incomplet (5 Mo = suspect).")
    print(f"  Periode : {str(time[0])[:10]} -> {str(time[-1])[:10]}  "
          f"({len(time)} pas de temps)")

    # Verifier qu on a bien des donnees mensuelles 1993-2023
    if len(time) < 100:
        print(f"  ATTENTION : seulement {len(time)} pas de temps. "
              f"Relancer le telechargement si < 372.")

    # Extraction aux stations
    # Verifier la plage de longitudes du fichier
    lon_vals = ds[lon_name].values if lon_name in ds.coords else msl[lon_name].values
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
    print(f"  Plage lon ERA5 : {lon_min:.1f} a {lon_max:.1f}")

    # San Fernando lon : -6.20 ou 353.8 selon convention
    lon_sf = -6.20 if lon_min < 0 else 360.0 - 6.20
    lon_pa = 11.87

    sel_sf = {lat_name: 36.45, lon_name: lon_sf}
    sel_pa = {lat_name: 45.37, lon_name: lon_pa}

    p_sf = msl.sel(sel_sf, method='nearest').values.flatten()
    p_pa = msl.sel(sel_pa, method='nearest').values.flatten()
    ds.close()

    # S assurer que p_sf et p_pa ont la meme longueur que time
    n = min(len(time), len(p_sf), len(p_pa))
    time = time[:n]; p_sf = p_sf[:n]; p_pa = p_pa[:n]

    print(f"  San Fernando ({lon_sf:.1f}E) : {p_sf.mean()/100:.1f} hPa")
    print(f"  Padua        ({lon_pa:.1f}E) : {p_pa.mean()/100:.1f} hPa")

    # Conversion Pa -> hPa
    if p_sf.mean() > 10000:
        p_sf = p_sf / 100.0
        p_pa = p_pa / 100.0

    # Normalisation mensuelle
    months = np.array([int(str(np.datetime64(t, 'M'))[-2:]) for t in time])
    p_sf_std = np.zeros(n)
    p_pa_std = np.zeros(n)

    for m in range(1, 13):
        idx = months == m
        if idx.sum() == 0:
            continue
        p_sf_std[idx] = (p_sf[idx] - p_sf[idx].mean()) / (p_sf[idx].std() + 1e-12)
        p_pa_std[idx] = (p_pa[idx] - p_pa[idx].mean()) / (p_pa[idx].std() + 1e-12)

    # Indice WeMO
    # v1.1: canonical sign, San Fernando minus Padua
    # (Martin-Vide and Lopez-Bustins, 2006); v1.0 computed the opposite.
    wemo     = p_sf_std - p_pa_std
    wemo_std = (wemo - wemo.mean()) / (wemo.std() + 1e-12)

    print(f"  WeMO calcule : {n} mois  mean={wemo_std.mean():.4f}  std={wemo_std.std():.4f}")
    return wemo_std, time



# ÉTAPE 3 — EKE TWO-SAT + CORRÉLATION CROISÉE
def compute_eke_monthly(sla_file):
    """Charge EKE depuis DUACS TWO-SAT, resample mensuel, basin mean."""
    print("\nChargement EKE TWO-SAT...")
    ds = xr.open_dataset(sla_file)
    ugos = ds['ugos'].resample(time='MS').mean()
    vgos = ds['vgos'].resample(time='MS').mean()
    ekt  = 0.5 * (ugos**2 + vgos**2) * 1e4
    eke_basin = ekt.mean(dim=['latitude','longitude']).values
    time_eke  = ekt.time.values
    ds.close()

    # Désaisonnalisation
    months = np.array([int(str(t)[5:7]) for t in time_eke])
    eke_anom = np.zeros_like(eke_basin)
    for m in range(1, 13):
        idx = months == m
        eke_anom[idx] = eke_basin[idx] - eke_basin[idx].mean()

    print(f"  EKE TWO-SAT : {len(eke_anom)} mois")
    return eke_anom, time_eke


def cross_correlation_analysis(wemo, time_wemo, eke, time_eke, max_lag=24):
    """Corrélation croisée WeMO–EKE sur période commune."""
    print("\nCorrélation croisée WeMO–EKE...")

    t_min = max(time_wemo[0], time_eke[0])
    t_max = min(time_wemo[-1], time_eke[-1])
    mask_w = (time_wemo >= t_min) & (time_wemo <= t_max)
    mask_e = (time_eke  >= t_min) & (time_eke  <= t_max)
    w = wemo[mask_w]
    e = eke[mask_e]
    n = min(len(w), len(e))
    w = w[:n]; e = e[:n]

    lags   = np.arange(-max_lag, max_lag+1)
    corrs  = np.zeros(len(lags))
    pvals  = np.zeros(len(lags))

    for k, lag in enumerate(lags):
        if lag == 0:
            r, p = stats.pearsonr(w, e)
        elif lag > 0:
            r, p = stats.pearsonr(w[:-lag], e[lag:])
        else:
            r, p = stats.pearsonr(w[-lag:], e[:lag])
        corrs[k] = r
        pvals[k] = p

    i_opt   = np.argmax(np.abs(corrs))
    lag_opt = lags[i_opt]
    r_opt   = corrs[i_opt]
    p_opt   = pvals[i_opt]
    r_zero  = corrs[lags == 0][0]
    p_zero  = pvals[lags == 0][0]
    sig_lags = lags[pvals < 0.05]

    print(f"\n  == RÉSULTATS CORRÉLATION CROISÉE WeMO–EKE ==")
    print(f"  Période    : {str(t_min)[:7]} -> {str(t_max)[:7]}  (n={n})")
    print(f"  lag=0      : r = {r_zero:+.3f}  p = {p_zero:.5f}")
    print(f"  Lag optimal: lag = {lag_opt:+d} mois")
    print(f"  r(lag_opt) : r = {r_opt:+.3f}  p = {p_opt:.5f}")
    print(f"  Lags sig.  : {sig_lags[0]:+d} à {sig_lags[-1]:+d} mois")
    print(f"  =============================================")

    return lags, corrs, pvals, lag_opt, r_opt, p_opt, r_zero, p_zero, n, sig_lags


# ÉTAPE 4 — FIGURE PUBLICATION-READY
def make_wemo_figure(lags, corrs, pvals, wemo, time_wemo,
                     eke, time_eke, lag_opt, r_opt, p_opt, n, out_fig):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=300)
    fig.patch.set_facecolor('white')

    # Panneau 1 : séries temporelles
    years_w = np.array([1993 + i/12 for i in range(len(wemo))])
    years_e = np.array([1993 + i/12 for i in range(len(eke))])
    ax1b = ax1.twinx()
    l1, = ax1.plot(years_w, wemo, color='#185FA5', lw=0.8, alpha=0.65,
                   label='WeMO (ERA5 reconstruction)')
    ax1.fill_between(years_w, wemo, 0, where=wemo < 0,
                     color='#185FA5', alpha=0.12)
    ax1.axhline(0, color='gray', lw=0.5, ls='--')
    ax1.set_ylabel('WeMO (standardised)', color='#185FA5', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='#185FA5', labelsize=8)

    l2, = ax1b.plot(years_e, eke, color='#D85A30', lw=0.9, alpha=0.85,
                    label='EKE anomaly (TWO-SAT)')
    ax1b.axhline(0, color='#D85A30', lw=0.4, ls=':')
    ax1b.set_ylabel('EKE anomaly [cm² s⁻²]', color='#D85A30', fontsize=9)
    ax1b.tick_params(axis='y', labelcolor='#D85A30', labelsize=8)
    ax1.set_xlim(1993, 2024)
    ax1.set_xlabel('Year', fontsize=9)
    ax1.legend(handles=[l1, l2], loc='upper left', fontsize=8, framealpha=0.9)
    ax1.set_title(
        f'(a) WeMO index (ERA5) and basin-averaged EKE anomaly, 1993–2023  (n={n})',
        fontsize=9, loc='left', fontweight='bold')
    # ligne tiretée à 2013 pour montrer l'extension par rapport à la période publiée
    ax1.axvline(2013.5, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax1.text(2010.5, ax1.get_ylim()[1]*0.88, 'Published\nrecord',
             fontsize=7, color='gray', ha='center')
    ax1.text(2017, ax1.get_ylim()[1]*0.88, 'ERA5\nextension',
             fontsize=7, color='#185FA5', ha='center')

    # Panneau 2 : corrélogramme
    sig = pvals < 0.05
    colors = ['#185FA5' if (s and c < 0) else '#D85A30' if (s and c > 0)
              else '#cccccc' for s, c in zip(sig, corrs)]
    ax2.bar(lags, corrs, width=0.8, color=colors, alpha=0.8, edgecolor='none')
    thresh = 1.96 / np.sqrt(n)
    ax2.axhline( thresh, color='#555555', lw=0.7, ls='--', alpha=0.7,
                label=f'p = 0.05 (±{thresh:.3f})')
    ax2.axhline(-thresh, color='#555555', lw=0.7, ls='--', alpha=0.7)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.axvline(lag_opt, color='#185FA5', lw=1.2, ls='-', alpha=0.8)
    ax2.annotate(
        f'lag = {lag_opt:+d} months\nr = {r_opt:.3f},  p = {p_opt:.4f}',
        xy=(lag_opt, r_opt),
        xytext=(lag_opt + (4 if lag_opt <= 0 else -12), r_opt * 0.55),
        fontsize=8.5, color='#185FA5',
        arrowprops=dict(arrowstyle='->', color='#185FA5', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                  ec='#185FA5', alpha=0.92, lw=0.6)
    )
    ax2.set_xlabel('Lag (months; positive = WeMO leads EKE)', fontsize=9)
    ax2.set_ylabel('Pearson r', fontsize=9)
    ax2.set_xlim(-max(abs(lags))-0.5, max(abs(lags))+0.5)
    ax2.set_ylim(-0.42, 0.42)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_title(
        f'(b) WeMO–EKE cross-correlation (1993–2023, n={n})',
        fontsize=9, loc='left', fontweight='bold')
    ax2.tick_params(labelsize=8)

    plt.tight_layout(pad=1.8)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white')
    # PNG aussi pour les slides
    fig.savefig(out_fig.replace('.pdf', '.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    print(f"\nFigure sauvegardée : {out_fig}")
    plt.close()


# MAIN
if __name__ == '__main__':
    print("=" * 60)
    print("EXTENSION WeMO 1993–2023 — ERA5 RECONSTRUCTION")
    print("=" * 60)

    # Téléchargement ERA5 si nécessaire
    if not os.path.exists(ERA5_F):
        print(f"\nERA5 non trouvé : {ERA5_F}")
        print("Lancement téléchargement CDS (compte requis)...")
        download_era5_mslp()
    else:
        print(f"\nERA5 déjà présent ({os.path.getsize(ERA5_F)/1e6:.0f} Mo)")

    # Calcul WeMO
    wemo, time_wemo = compute_wemo_index(ERA5_F)

    # Sauvegarde CSV
    import csv
    with open(OUT_CSV, 'w', newline='') as f:
        w_csv = csv.writer(f)
        w_csv.writerow(['time', 'wemo_era5'])
        for t, v in zip(time_wemo, wemo):
            w_csv.writerow([str(t)[:7], f'{v:.4f}'])
    print(f"CSV : {OUT_CSV}")

    # EKE TWO-SAT
    eke, time_eke = compute_eke_monthly(EKT_F)

    # Corrélation croisée
    lags, corrs, pvals, lag_opt, r_opt, p_opt, r_zero, p_zero, n, sig_lags = \
        cross_correlation_analysis(wemo, time_wemo, eke, time_eke)

    # Figure
    make_wemo_figure(lags, corrs, pvals, wemo, time_wemo,
                     eke, time_eke, lag_opt, r_opt, p_opt, n, OUT_FIG)

