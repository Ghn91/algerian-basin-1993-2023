# Animation of the Algerian Basin surface fields
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

"""
ANIMATION BASSIN ALGÉRIEN — VERSION FINALE CORRIGÉE
Pas de dask · Variables exactes certifiées par diagnostic
Nedjar, Bachari, Houma — Ocean Science (EGU)

Exécuter dans : conda activate eddy_env
Commande     : python animation_v3_final.py
"""

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
# FICHIERS — noms exacts confirmés par diagnostic
# ════════════════════════════════════════════════════════════════════

F_TWOSAT = BASE + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc"
F_SST    = BASE + r"\cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_multi-vars_4.96W-9.95E_35.04N-40.96N_1993-01-01-2023-12-31.nc"
F_CHL    = BASE + r"\cmems_mod_med_bgc-plankton_my_4.2km_P1D-m_chl-phyc_5.00W-10.00E_35.02N-40.98N_1.02m_1999-01-01-2023-12-31.nc"
F_TEMP   = BASE + r"\cmems_mod_med_phy-temp_my_4.2km_P1D-m_thetao_5.00W-10.00E_35.02N-40.98N_1.02m_1993-01-01-2023-12-31.nc"
F_MLD    = BASE + r"\cmems_mod_med_phy-mld_my_4.2km_P1D-m_mlotst_5.00W-10.00E_35.02N-40.98N_1993-01-01-2023-12-31.nc"

OUT_MP4 = BASE + r"\FIGURES\animation_bassin_algerien_v3.mp4"
OUT_GIF = BASE + r"\FIGURES\animation_bassin_algerien_v3.gif"

# ════════════════════════════════════════════════════════════════════
# PARAMÈTRES — ajuster selon votre PC
# ════════════════════════════════════════════════════════════════════
DPI    = 150    # 150=qualité pub · 100=rapide preview
FPS    = 20
SMOOTH = 0.8    # lissage gaussien (pixels)

print("=" * 65)
print("ANIMATION BASSIN ALGÉRIEN — chargement et rééchantillonnage")
print("=" * 65)

# ════════════════════════════════════════════════════════════════════
# [1] DUACS TWO-SAT — SLA + ugos + vgos
#     Variables confirmées : adt, sla, ugos, vgos, tpa_correction
#     Dims : (time=11322, latitude=24, longitude=60)
#     Période : 1993-01-01 → 2023-12-31
# ════════════════════════════════════════════════════════════════════
print("\n[1/4] DUACS TWO-SAT — SLA + vitesses (rééchantillonnage mensuel...)")
ds_sl = xr.open_dataset(F_TWOSAT)   # PAS de chunks — dask absent

sla_daily  = ds_sl['sla']           # (11322, 24, 60) en mètres
ugos_daily = ds_sl['ugos']          # (11322, 24, 60) en m/s
vgos_daily = ds_sl['vgos']          # (11322, 24, 60) en m/s

# Rééchantillonnage mensuel — réduit 11322 → ~372 pas de temps
sla_m  = sla_daily.resample(time='MS').mean()
ugos_m = ugos_daily.resample(time='MS').mean()
vgos_m = vgos_daily.resample(time='MS').mean()

# Calcul EKT [cm²/s²] et vitesse [cm/s]
ekt_m  = 0.5 * (ugos_m**2 + vgos_m**2) * 1e4
spd_m  = np.sqrt(ugos_m**2 + vgos_m**2) * 100.0

# Chargement en mémoire numpy (rapide après resample)
print("  Chargement en mémoire numpy...")
sla_arr  = sla_m.values    # (N, 24, 60)
ugos_arr = ugos_m.values
vgos_arr = vgos_m.values
ekt_arr  = ekt_m.values
spd_arr  = spd_m.values
time_arr = sla_m.time.values
lon_sl   = sla_m.longitude.values   # (60,)
lat_sl   = sla_m.latitude.values    # (24,)
N        = len(time_arr)

ds_sl.close()
print(f"  OK — {N} mois  ({str(time_arr[0])[:7]} → {str(time_arr[-1])[:7]})")
print(f"  EKT : moy={np.nanmean(ekt_arr):.1f}  max97={np.nanpercentile(ekt_arr,97):.1f} cm²/s²")

ekt_vmax = float(np.nanpercentile(ekt_arr, 97))
sla_lim  = float(np.nanpercentile(np.abs(sla_arr), 96))

# ════════════════════════════════════════════════════════════════════
# [2] SST OBSERVÉE L4 — analysed_sst en Kelvin
#     Dims : (time=11322, latitude=119, longitude=299)
#     Grille plus fine → on la rééchantillonne aussi
# ════════════════════════════════════════════════════════════════════
print("\n[2/4] SST L4 OBSERVATIONS (rééchantillonnage mensuel...)")
ds_sst = xr.open_dataset(F_SST)
sst_daily = ds_sst['analysed_sst']          # Kelvin
sst_m_K   = sst_daily.resample(time='MS').mean()
sst_m_C   = sst_m_K - 273.15               # → Celsius
# Anomalie par rapport à la climatologie mensuelle
sst_clim  = sst_m_C.groupby('time.month').mean('time')
sst_anom  = sst_m_C.groupby('time.month') - sst_clim

sst_arr  = sst_anom.values    # (N, 119, 299)
lon_sst  = sst_m_C.longitude.values
lat_sst  = sst_m_C.latitude.values
sst_lim  = float(np.nanpercentile(np.abs(sst_arr), 96))

ds_sst.close()
print(f"  OK — anomalies SST ±{sst_lim:.2f}°C")

# ════════════════════════════════════════════════════════════════════
# [3] CHLOROPHYLLE-a
#     Dims : (time=9131, depth=1, latitude=144, longitude=361)
#     Période : 1999–2023 → aligner sur time_arr
# ════════════════════════════════════════════════════════════════════
print("\n[3/4] Chlorophylle-a BGC (1999–2023)...")
ds_chl  = xr.open_dataset(F_CHL)
chl_daily = ds_chl['chl'].isel(depth=0)    # supprimer dim depth
chl_m     = chl_daily.resample(time='MS').mean()
chl_arr_raw = chl_m.values                 # (N_chl, 144, 361)
time_chl    = chl_m.time.values
lon_chl     = chl_m.longitude.values
lat_chl     = chl_m.latitude.values
ds_chl.close()
print(f"  OK — {len(time_chl)} mois Chl-a  [{str(time_chl[0])[:7]} → {str(time_chl[-1])[:7]}]")

# Index de correspondance chl ↔ time_arr (pour les frames post-1999)
def get_chl_idx(t):
    """Retourne l'index chl pour un timestamp donné, ou None."""
    if t < time_chl[0] or t > time_chl[-1]:
        return None
    diffs = np.abs((time_chl - t).astype('int64'))
    return int(np.argmin(diffs))

# ════════════════════════════════════════════════════════════════════
# [4] TEMPÉRATURE RÉANALYSE — thetao (°C)
#     Dims : (time=11322, depth=1, latitude=144, longitude=361)
# ════════════════════════════════════════════════════════════════════
print("\n[4/4] Température réanalyse (thetao)...")
ds_temp   = xr.open_dataset(F_TEMP)
temp_daily = ds_temp['thetao'].isel(depth=0)
temp_m     = temp_daily.resample(time='MS').mean()
temp_arr   = temp_m.values                 # (N, 144, 361) en °C
lon_temp   = temp_m.longitude.values
lat_temp   = temp_m.latitude.values
ds_temp.close()
print(f"  OK — thetao °C  shape={temp_arr.shape}")

# ════════════════════════════════════════════════════════════════════
# GRILLES 2D
# ════════════════════════════════════════════════════════════════════
LON2D_sl,  LAT2D_sl  = np.meshgrid(lon_sl,  lat_sl)
LON2D_sst, LAT2D_sst = np.meshgrid(lon_sst, lat_sst)
LON2D_chl, LAT2D_chl = np.meshgrid(lon_chl, lat_chl)
LON2D_tmp, LAT2D_tmp = np.meshgrid(lon_temp, lat_temp)

# ════════════════════════════════════════════════════════════════════
# CONSTRUCTION FIGURE — style WHOI/Copernicus
# ════════════════════════════════════════════════════════════════════
proj = ccrs.PlateCarree()
fig  = plt.figure(figsize=(18, 11), dpi=DPI)
fig.patch.set_facecolor('#03101f')

# Carte principale (75% largeur)
ax = fig.add_axes([0.02, 0.07, 0.71, 0.90], projection=proj)
ax.set_facecolor('#03101f')
ax.set_extent([-2.5, 9.5, 34.8, 41.8], crs=proj)

# Panneau droite haut — EKT time series
ax_ekt = fig.add_axes([0.75, 0.70, 0.23, 0.24])
ax_ekt.set_facecolor('#020c1e')
for sp in ax_ekt.spines.values(): sp.set_edgecolor('#152540')
ax_ekt.tick_params(colors='#4a8aaa', labelsize=6)

# Panneau droite milieu — Chl bar
ax_chl = fig.add_axes([0.75, 0.42, 0.23, 0.24])
ax_chl.set_facecolor('#020c1e')
for sp in ax_chl.spines.values(): sp.set_edgecolor('#152540')
ax_chl.tick_params(colors='#4a8aaa', labelsize=6)

# Panneau droite bas — stats
ax_stat = fig.add_axes([0.75, 0.07, 0.23, 0.30])
ax_stat.set_facecolor('#020c1e')
ax_stat.set_xticks([]); ax_stat.set_yticks([])
for sp in ax_stat.spines.values(): sp.set_edgecolor('#152540')

# ─── Fond cartographique ──────────────────────────────────────────
ax.add_feature(cfeature.LAND.with_scale('10m'),
               facecolor='#0d1a0d', edgecolor='#1c2c1c', linewidth=0.4, zorder=8)
ax.add_feature(cfeature.COASTLINE.with_scale('10m'),
               edgecolor='#2a4a2a', linewidth=0.8, zorder=9)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),
               edgecolor='#152515', linewidth=0.3, linestyle=':', zorder=9)

gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='#152540',
                  alpha=0.6, linestyle='--', zorder=7)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {'size': 7.5, 'color': '#4a8aaa'}
gl.xlocator = mticker.FixedLocator([-2, 0, 2, 4, 6, 8])
gl.ylocator = mticker.FixedLocator([35, 36, 37, 38, 39, 40, 41])

# ─── Couche 1 : SST anomalie (fond riche) ─────────────────────────
sst0 = gaussian_filter(np.nan_to_num(sst_arr[0]), sigma=SMOOTH*2)
pcm_sst = ax.pcolormesh(LON2D_sst, LAT2D_sst, sst0,
                         cmap=cmocean.cm.balance,
                         vmin=-sst_lim, vmax=sst_lim,
                         transform=proj, zorder=1, shading='gouraud', alpha=0.82)

# ─── Couche 2 : EKT (overlay énergétique) ─────────────────────────
ekt0 = gaussian_filter(np.nan_to_num(ekt_arr[0]), sigma=SMOOTH)
pcm_ekt = ax.pcolormesh(LON2D_sl, LAT2D_sl, ekt0,
                         cmap=cmocean.cm.thermal,
                         vmin=0, vmax=ekt_vmax,
                         transform=proj, zorder=2, shading='gouraud', alpha=0.52)

# ─── Couche 3 : Contours SLA (topographie dynamique) ──────────────
sla0_sm = gaussian_filter(np.nan_to_num(sla_arr[0]), sigma=SMOOTH)
cs_sla = [ax.contour(LON2D_sl, LAT2D_sl, sla0_sm,
                      levels=12, cmap='coolwarm',
                      linewidths=0.5, alpha=0.55,
                      transform=proj, zorder=3)]

# ─── Couche 4 : Streamlines (courant algérien) ────────────────────
step = 2   # sous-échantillonnage grille 0.25° → step=2 OK
spd_vmax = float(np.nanpercentile(spd_arr, 97))  # calculé ici, utilisé partout

def norm_speed(s):
    """Normalise un tableau de vitesse entre 0 et 1 pour streamplot."""
    s_clip = np.clip(np.nan_to_num(s), 0, spd_vmax)
    return s_clip / spd_vmax if spd_vmax > 0 else s_clip

u0 = np.nan_to_num(ugos_arr[0])
v0 = np.nan_to_num(vgos_arr[0])
s0 = gaussian_filter(np.nan_to_num(spd_arr[0]), sigma=SMOOTH)
stream_obj = [ax.streamplot(
    lon_sl[::step], lat_sl[::step],
    u0[::step, ::step], v0[::step, ::step],
    density=2.2, linewidth=0.9,
    color=norm_speed(s0[::step, ::step]),
    cmap=cmocean.cm.speed,
    arrowsize=0.8, arrowstyle='->',
    transform=proj, zorder=4)]

# ─── Couche 5 : Chl-a (overlay vert si disponible) ────────────────
chl_idx0 = get_chl_idx(time_arr[0])
chl_obj = [None]
if chl_idx0 is not None:
    chl0 = np.log1p(gaussian_filter(np.nan_to_num(chl_arr_raw[chl_idx0]), sigma=SMOOTH*1.5))
    chl_max = float(np.nanpercentile(np.log1p(chl_arr_raw), 97))
    chl_obj[0] = ax.pcolormesh(LON2D_chl, LAT2D_chl, chl0,
                                cmap=cmocean.cm.algae,
                                vmin=0, vmax=chl_max,
                                transform=proj, zorder=5,
                                shading='gouraud', alpha=0.28)

# ─── Colorbars ────────────────────────────────────────────────────
cax1 = fig.add_axes([0.73, 0.55, 0.012, 0.40])
cb1  = plt.colorbar(pcm_ekt, cax=cax1)
cb1.set_label('EKT [cm²/s²]', color='#8aaacc', fontsize=8, labelpad=3)
cb1.ax.yaxis.set_tick_params(color='#4a8aaa', labelsize=5)
plt.setp(cb1.ax.yaxis.get_ticklabels(), color='#4a8aaa')
cb1.outline.set_edgecolor('#152540')

cax2 = fig.add_axes([0.73, 0.10, 0.012, 0.40])
cb2  = plt.colorbar(pcm_sst, cax=cax2)
cb2.set_label('SST anom. [°C]', color='#8aaacc', fontsize=8, labelpad=3)
cb2.ax.yaxis.set_tick_params(color='#4a8aaa', labelsize=5)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='#4a8aaa')
cb2.outline.set_edgecolor('#152540')

# ─── Textes fixes ─────────────────────────────────────────────────
title_main = ax.text(0.5, 1.010,
    'Bassin Algérien — Dynamique méso-échelle de surface',
    transform=ax.transAxes, color='#c0d8f0',
    fontsize=13, fontweight='bold', ha='center', va='bottom')

date_txt = ax.text(0.98, 1.010, '',
    transform=ax.transAxes, color='#EF9F27',
    fontsize=12, fontweight='bold', ha='right', va='bottom',
    fontfamily='monospace')

ekt_live = ax.text(0.01, 0.99, '',
    transform=ax.transAxes, color='#5DCAA5',
    fontsize=9, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.35', fc='#03101f',
              ec='#5DCAA5', alpha=0.85, lw=0.6))

fig.text(0.50, 0.005,
    'Nedjar · Bachari · Houma  |  Ocean Science (EGU)  |  '
    'Données CMEMS DUACS TWO-SAT · SST L4 OBS · BGC réanalyse  |  1993–2023',
    ha='center', fontsize=6.5, color='#253545', style='italic')

# ─── Panneau EKT time series ──────────────────────────────────────
ekt_basin = [float(np.nanmean(ekt_arr[i])) for i in range(N)]
years_f   = [1993 + i/12 for i in range(N)]

ax_ekt.set_xlim(1993, 2024)
ax_ekt.set_ylim(0, max(ekt_basin) * 1.22)
ax_ekt.set_title('EKT bassin moyen [cm²/s²]', color='#8aaacc', fontsize=7.5, pad=2)
ax_ekt.set_xlabel('Année', color='#4a8aaa', fontsize=6)

# Tendance statique
tx = np.array([1993., 2023.])
ty = np.array([52., 52. + 2.06 * 3.0])
ax_ekt.plot(tx, ty, '--', color='#EF9F27', lw=1.2, alpha=0.75, label='+2.06 cm²/s²/déc')
ax_ekt.axvline(2005.5, color='#D85A30', lw=0.6, linestyle=':', alpha=0.6)
ax_ekt.text(2005.8, max(ekt_basin)*1.12, 'Rupture\n2005', color='#D85A30', fontsize=5.5)
ax_ekt.legend(loc='upper left', fontsize=5.5, facecolor='#020c1e',
              edgecolor='#152540', labelcolor='#8aaacc', framealpha=0.9)

ts_fill  = ax_ekt.fill_between([], [], alpha=0.15, color='#5DCAA5')
ts_line, = ax_ekt.plot([], [], color='#5DCAA5', lw=1.0)
ts_dot,  = ax_ekt.plot([], [], 'o', color='#EF9F27', ms=4.5, zorder=5)

# ─── Panneau Chl-a time series ────────────────────────────────────
chl_basin_ts = {}
for ci, t in enumerate(time_chl):
    # Trouver l'index dans time_arr
    diffs = np.abs((time_arr - t).astype('int64'))
    idx_in_main = int(np.argmin(diffs))
    chl_basin_ts[idx_in_main] = float(np.nanmean(chl_arr_raw[ci]))

chl_x = sorted(chl_basin_ts.keys())
chl_y = [chl_basin_ts[k] for k in chl_x]
chl_years = [years_f[k] for k in chl_x]

ax_chl.set_xlim(1993, 2024)
ax_chl.set_ylim(0, max(chl_y)*1.25 if chl_y else 1)
ax_chl.set_title('Chl-a moy. bassin [mg/m³]', color='#8aaacc', fontsize=7.5, pad=2)
ax_chl.plot(chl_years, chl_y, color='#1D9E75', lw=0.6, alpha=0.4)
chl_line, = ax_chl.plot([], [], color='#5DCAA5', lw=1.0)
chl_dot,  = ax_chl.plot([], [], 'o', color='#5DCAA5', ms=4, zorder=5)
ax_chl.tick_params(colors='#4a8aaa', labelsize=6)
for sp in ax_chl.spines.values(): sp.set_edgecolor('#152540')

# ─── Panneau stats fixes ──────────────────────────────────────────
stats_data = [
    ('─── EKT',           '',                    '#5DCAA5',  True),
    ('Tendance',          '+2.06 cm²/s²/déc',    '#EF9F27',  False),
    ('p-value',           '0.004',               '#EF9F27',  False),
    ('Rupture Pettitt',   '2005',                '#D85A30',  False),
    ('Wavelet dominante', '74 mois / 6.2 ans',   '#AFA9EC',  False),
    ('',                  '',                    '',         False),
    ('─── FORÇAGE',       '',                    '#85B7EB',  True),
    ('WeMO r',            '−0.218  (p=0.0005)',  '#85B7EB',  False),
    ('WeMO lag',          '−2 mois',             '#85B7EB',  False),
    ('NAO r',             '+0.038  (NS)',         '#444461',  False),
    ('',                  '',                    '',         False),
    ('─── EOF',           '',                    '#9FE1CB',  True),
    ('EOF1',              '54.2 %',              '#9FE1CB',  False),
    ('4 modes',           '66.2 %',              '#9FE1CB',  False),
    ('',                  '',                    '',         False),
    ('─── EDDIES',        '',                    '#F0997B',  True),
    ('Anticyclones',      '7,215 obs.',          '#F0997B',  False),
    ('Max AC (2023)',      '313 obs.',            '#F0997B',  False),
    ('Cyclones',          '4,525 obs.',          '#85B7EB',  False),
    ('Vitesse AC/CY',     '32.7 / 13.7 cm/s',   '#D3D1C7',  False),
]
for idx, (lbl, val, col, bold) in enumerate(stats_data):
    y = 0.97 - idx * 0.048
    fw = 'bold' if bold else 'normal'
    ax_stat.text(0.04, y, lbl, transform=ax_stat.transAxes,
                 color=col if col else '#4a6a8a',
                 fontsize=7.0, fontweight=fw)
    if val:
        ax_stat.text(0.97, y, val, transform=ax_stat.transAxes,
                     color=col, fontsize=7.0, ha='right', fontweight='bold')

# ════════════════════════════════════════════════════════════════════
# FONCTION UPDATE — le cœur de l'animation
# ════════════════════════════════════════════════════════════════════
chl_logmax = float(np.nanpercentile(np.log1p(chl_arr_raw[~np.isnan(chl_arr_raw)]), 97))

def update(frame):
    i = frame % N
    yr = years_f[i]
    date_str = str(time_arr[i])[:7]
    ekt_mean = ekt_basin[i]

    # ── Titre & stats live ──────────────────────────────────────────
    date_txt.set_text(date_str)
    ekt_live.set_text(
        f'EKT bassin : {ekt_mean:.1f} cm²/s²\n'
        f'SLA moy.   : {float(np.nanmean(sla_arr[i]))*100:.1f} cm')

    # ── SST anomalie ────────────────────────────────────────────────
    sst_i = gaussian_filter(np.nan_to_num(sst_arr[i]), sigma=SMOOTH * 2.0)
    pcm_sst.set_array(sst_i.ravel())

    # ── EKT ─────────────────────────────────────────────────────────
    ekt_i = gaussian_filter(np.nan_to_num(ekt_arr[i]), sigma=SMOOTH)
    pcm_ekt.set_array(ekt_i.ravel())

    # ── Contours SLA ────────────────────────────────────────────────
    for coll in cs_sla[0].collections:
        try: coll.remove()
        except: pass
    sla_i = gaussian_filter(np.nan_to_num(sla_arr[i]), sigma=SMOOTH)
    cs_sla[0] = ax.contour(LON2D_sl, LAT2D_sl, sla_i,
                            levels=12, cmap='coolwarm',
                            linewidths=0.5, alpha=0.55,
                            transform=proj, zorder=3)

    # ── Streamlines ─────────────────────────────────────────────────
    try:
        stream_obj[0].lines.remove()
        for arr in stream_obj[0].arrows.patches:
            arr.remove()
    except: pass
    u_i = np.nan_to_num(ugos_arr[i])
    v_i = np.nan_to_num(vgos_arr[i])
    s_i = gaussian_filter(np.nan_to_num(spd_arr[i]), sigma=SMOOTH)
    stream_obj[0] = ax.streamplot(
        lon_sl[::step], lat_sl[::step],
        u_i[::step, ::step], v_i[::step, ::step],
        density=2.2, linewidth=0.9,
        color=norm_speed(s_i[::step, ::step]),
        cmap=cmocean.cm.speed,
        arrowsize=0.8, arrowstyle='->',
        transform=proj, zorder=4)

    # ── Chl-a (si disponible pour cette date) ───────────────────────
    ci = get_chl_idx(time_arr[i])
    if ci is not None and chl_obj[0] is not None:
        chl_i = np.log1p(gaussian_filter(
            np.nan_to_num(chl_arr_raw[ci]), sigma=SMOOTH * 1.5))
        chl_obj[0].set_array(chl_i.ravel())

    # ── Time series EKT ─────────────────────────────────────────────
    ts_line.set_data(years_f[:i+1], ekt_basin[:i+1])
    ts_dot.set_data([yr], [ekt_mean])

    # ── Time series Chl ─────────────────────────────────────────────
    chl_x_now = [years_f[k] for k in chl_x if k <= i]
    chl_y_now = [chl_basin_ts[k] for k in chl_x if k <= i]
    chl_line.set_data(chl_x_now, chl_y_now)
    if chl_x_now:
        chl_dot.set_data([chl_x_now[-1]], [chl_y_now[-1]])

    if frame % 6 == 0:
        print(f"  [{frame+1:3d}/{N}]  {date_str}  "
              f"EKT={ekt_mean:5.1f}  "
              f"SLA={float(np.nanmean(sla_arr[i]))*100:+4.1f} cm")

    return [pcm_sst, pcm_ekt, ts_line, ts_dot, ekt_live, date_txt]


# ════════════════════════════════════════════════════════════════════
# RENDU — MP4 puis GIF
# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"RENDU MP4 : {N} frames · {DPI} dpi · {FPS} fps")
print(f"Sortie    : {OUT_MP4}")
print(f"Durée est.: {N * 10 // 60}–{N * 20 // 60} minutes selon votre PC")
print(f"{'='*65}\n")

# ─── Vérification ffmpeg ──────────────────────────────────────────
import shutil, subprocess, os

ffmpeg_ok = shutil.which('ffmpeg') is not None
if not ffmpeg_ok:
    # Chercher dans l'environnement conda
    conda_ffmpeg = os.path.join(
        os.path.dirname(shutil.which('python') or ''),
        'ffmpeg.exe')
    ffmpeg_ok = os.path.isfile(conda_ffmpeg)
    if ffmpeg_ok:
        matplotlib.rcParams['animation.ffmpeg_path'] = conda_ffmpeg

if ffmpeg_ok:
    print("ffmpeg trouvé — rendu MP4 en cours...")
    ani = animation.FuncAnimation(
        fig, update, frames=N, interval=1000 // FPS, blit=False)
    Writer = animation.FFMpegWriter(
        fps=FPS, bitrate=6000,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '16'])
    ani.save(OUT_MP4, writer=Writer, dpi=DPI)
    print(f"\nMP4 sauvegardé : {OUT_MP4}")
else:
    print("ATTENTION : ffmpeg introuvable.")
    print("Installez-le avec : conda install -c conda-forge ffmpeg")
    print("En attendant → rendu GIF direct avec Pillow...\n")
    ani = animation.FuncAnimation(
        fig, update, frames=N, interval=1000 // FPS, blit=False)

# ─── GIF Pillow (fonctionne toujours, sans ffmpeg) ───────────────
print("Génération GIF (1 frame sur 3)...")
from PIL import Image
import io
os.makedirs(os.path.dirname(OUT_GIF), exist_ok=True)
frames_pil = []
for fi in range(0, N, 3):
    update(fi)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    frames_pil.append(Image.open(buf).copy())
    buf.close()
    if fi % 30 == 0:
        print(f"  GIF frame {fi}/{N}")

frames_pil[0].save(
    OUT_GIF, save_all=True, append_images=frames_pil[1:],
    loop=0, duration=130, optimize=True)
print(f"GIF sauvegardé : {OUT_GIF}")

plt.close()
print("\nTerminé.")
if ffmpeg_ok:
    print(f"  MP4 : {OUT_MP4}")
print(f"  GIF : {OUT_GIF}")
