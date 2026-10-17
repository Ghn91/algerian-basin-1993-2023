"""
Figure 5 - EOF decomposition of monthly SLA anomaly (ALL-SAT).

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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
from eofs.xarray import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'savefig.dpi':      300,
    'lines.linewidth':   0.8,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

FIG_DIR  = DATA_DIR + r"\FIGURES\PUBLICATION"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Chargement + préparation SLA ──────────────────────────────────────────
print("Chargement SLA allsat...")
ds = xr.open_dataset(
    DATA_DIR + r"\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg"
               r"_P1D_multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)
sla_monthly = ds['sla'].resample(time='ME').mean()
sla_clim    = sla_monthly.groupby('time.month').mean()
sla_anom    = sla_monthly.groupby('time.month') - sla_clim
sla_filled  = sla_anom.fillna(0.0)
ds.close()

# ── EOF ───────────────────────────────────────────────────────────────────
print("Calcul EOF...")
solver   = Eof(sla_filled, weights=None)
n_modes  = 4
eofs     = solver.eofsAsCovariance(neofs=n_modes)
pcs      = solver.pcs(npcs=n_modes, pcscaling=1)
var_frac = solver.varianceFraction()[:n_modes]
print("Variance expliquee :")
for i in range(n_modes):
    print(f"  EOF{i+1} : {float(var_frac[i])*100:.1f}%")

# ── Configuration figure ──────────────────────────────────────────────────
proj       = ccrs.PlateCarree()
EXTENT     = [-5, 10, 35, 41]
LAND_COLOR = '#DDD5C0'
OCEAN_BG   = '#EEF2F6'
LABEL_KW   = dict(fontsize=9, fontweight='bold', va='top', ha='left',
                  color='black', zorder=10,
                  bbox=dict(facecolor='white', alpha=0.82,
                            edgecolor='none', boxstyle='round,pad=0.2'))
GRID_KW    = dict(alpha=0.35, linestyle='--',
                  linewidth=0.35, color='#888888')

# Figure : 4 lignes × 2 colonnes
# Colonne gauche = carte EOF (géographique)
# Colonne droite = série temporelle PC
fig = plt.figure(figsize=(7.0, 8.8))
fig.patch.set_facecolor('white')

# GridSpec principal : 4 lignes, ratio largeur 45%/55%
gs = gridspec.GridSpec(
    4, 2,
    figure=fig,
    hspace=0.10,
    wspace=0.38,
    left=0.05,
    right=0.93,
    top=0.97,
    bottom=0.05,
    width_ratios=[1.05, 1.0]
)

# Palette divergente pour cartes EOF
CMAP_EOF = cmocean.cm.balance

# Labels panneaux
PANEL_LABELS_MAP = ['(a)', '(c)', '(e)', '(g)']
PANEL_LABELS_PC  = ['(b)', '(d)', '(f)', '(h)']

# Couleurs PC
C_POS  = '#D9534F'
C_NEG  = '#5B8DB8'
C_SMTH = '#1A1A2E'

def setup_geoax(ax, label):
    ax.set_extent(EXTENT, crs=proj)
    ax.set_facecolor(OCEAN_BG)
    ax.add_feature(cfeature.LAND,
                   facecolor=LAND_COLOR, zorder=5, linewidth=0)
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=0.45, color='#444444', zorder=6)
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.28, linestyle=':', color='#888888', zorder=6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.20,
                      color='#BBBBBB', alpha=0.7, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator([-4, 0, 4, 8])
    gl.ylocator     = mticker.FixedLocator([36, 38, 40])
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 5.5, 'color': '#333333'}
    gl.ylabel_style = {'size': 5.5, 'color': '#333333'}
    ax.text(0.014, 0.958, label, transform=ax.transAxes, **LABEL_KW)

def style_pc_ax(ax, label, xlabel=False):
    ax.grid(True, **GRID_KW)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.text(0.012, 0.965, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left',
            color='black', zorder=10,
            bbox=dict(facecolor='white', alpha=0.82,
                      edgecolor='none', boxstyle='round,pad=0.2'))
    if not xlabel:
        ax.tick_params(labelbottom=False)

# ── Stockage colorbars pour ajout après canvas.draw() ────────────────────
cbar_data = []

for mode in range(n_modes):
    var_pct  = float(var_frac[mode]) * 100
    row      = mode

    # ── Carte spatiale EOF (colonne gauche) ───────────────────────────────
    ax_map = fig.add_subplot(gs[row, 0], projection=proj)
    setup_geoax(ax_map, PANEL_LABELS_MAP[mode])
    ax_map.set_title(
        f'EOF {mode+1} — spatial pattern  ({var_pct:.1f}% variance)',
        fontsize=7, loc='center', pad=2.5, color='#111111'
    )

    eof_data = eofs.isel(mode=mode)
    vmax     = float(np.nanpercentile(np.abs(eof_data.values), 98))
    vmax     = max(vmax, 0.005)

    cf = ax_map.contourf(
        eof_data.longitude, eof_data.latitude, eof_data.values,
        levels=np.linspace(-vmax, vmax, 21),
        cmap=CMAP_EOF, transform=proj, zorder=2, extend='both'
    )
    # Ligne de zéro
    ax_map.contour(
        eof_data.longitude, eof_data.latitude, eof_data.values,
        levels=[0], colors='black', linewidths=0.6,
        transform=proj, zorder=3
    )
    cbar_data.append((cf, ax_map, f'Covariance (m)', '%.3f'))

    # ── Série temporelle PC (colonne droite) ──────────────────────────────
    ax_pc   = fig.add_subplot(gs[row, 1])
    is_last = (mode == n_modes - 1)
    style_pc_ax(ax_pc, PANEL_LABELS_PC[mode], xlabel=is_last)
    ax_pc.set_title(
        f'PC {mode+1}  ({var_pct:.1f}% variance)',
        fontsize=7, loc='center', pad=2.5, color='#111111'
    )

    pc_vals  = pcs.isel(mode=mode).values
    times_pc = pcs.time.values
    pc_sm    = uniform_filter1d(pc_vals, size=12, mode='nearest')

    ax_pc.fill_between(times_pc, pc_vals,
                       where=pc_vals >= 0,
                       color=C_POS, alpha=0.40, linewidth=0)
    ax_pc.fill_between(times_pc, pc_vals,
                       where=pc_vals < 0,
                       color=C_NEG, alpha=0.40, linewidth=0)
    ax_pc.plot(times_pc, pc_sm,
               color=C_SMTH, linewidth=1.5, zorder=4,
               label='12-month running mean')
    ax_pc.axhline(0, color='black', linewidth=0.55, zorder=3)

    # Rupture de régime sur EOF1 uniquement
    if mode == 0:
        ax_pc.axvline(np.datetime64('2008-01-01'),
                      color='#777777', linewidth=0.9,
                      linestyle=':', zorder=3)
        ax_pc.text(np.datetime64('2008-06-01'), ax_pc.get_ylim()[0]
                   if ax_pc.get_ylim()[0] != 0 else -2.8,
                   '~2008', fontsize=5.5, color='#555555',
                   va='bottom', ha='left')

    ax_pc.set_xlim(times_pc[0], times_pc[-1])
    ax_pc.set_ylabel('Amplitude (std)', fontsize=6.5)
    if is_last:
        ax_pc.set_xlabel('Year', fontsize=7)

    ax_pc.xaxis.set_major_locator(mdates.YearLocator(10))
    ax_pc.xaxis.set_minor_locator(mdates.YearLocator(5))
    ax_pc.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if mode == 0:
        leg = ax_pc.legend(
            loc='upper left', framealpha=0.85,
            edgecolor='#CCCCCC', fancybox=False,
            fontsize=6.0, handlelength=1.5,
            borderpad=0.4
        )
        leg.get_frame().set_linewidth(0.4)

# ── Colorbars après rendu ─────────────────────────────────────────────────
fig.canvas.draw()

for cf, ax, label, fmt in cbar_data:
    pos = ax.get_position()
    cax = fig.add_axes([
        pos.x1 + 0.008,
        pos.y0 + 0.008,
        0.011,
        pos.height - 0.016
    ])
    cb = fig.colorbar(cf, cax=cax, format=fmt)
    cb.set_label(label, fontsize=5.5, labelpad=2)
    cb.ax.tick_params(labelsize=5.0, width=0.4, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.locator = mticker.MaxNLocator(nbins=4, prune='both')
    cb.update_ticks()

# ── Sauvegarde ────────────────────────────────────────────────────────────
OUTPUT_PNG = FIG_DIR + r"\Figure_04_EOF_SLA.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_04_EOF_SLA.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()

print("""
CAPTION FIGURE 4:
Figure 4. Empirical orthogonal function (EOF) decomposition of monthly
sea level anomaly (SLA) in the Algerian Basin, 1993-2023 (CMEMS DUACS
DT-2024 ALL-SAT). Left column: spatial EOF patterns expressed as
covariance maps; the zero contour is shown as a black line. Right column:
corresponding normalised principal component (PC) time series; red and blue
shading indicate positive and negative phases respectively; the 12-month
running mean is shown in black.
(a-b) EOF1/PC1 (54.2% of variance): basin-wide coherent mode; PC1
documents a regime shift from persistently negative anomalies before 2008
(dotted line) to sustained positive values thereafter.
(c-d) EOF2/PC2 (5.0%): east-west dipole between the Alboran Sea and the
Algerian Basin interior, reflecting interannual modulation by the NAO.
(e-f) EOF3/PC3 (3.6%): mesoscale propagation mode associated with
eastward-propagating anticyclonic eddies along the Algerian margin.
(g-h) EOF4/PC4 (3.4%): coastal-offshore dipole linked to convergence
and divergence dynamics along the Algerian continental slope.
The four modes collectively account for 66.2% of total SLA variance.
""")
