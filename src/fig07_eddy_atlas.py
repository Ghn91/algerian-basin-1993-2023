"""
Figure 7 - mesoscale eddy atlas (py-eddy-tracker catalogue).

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

EDDY_DIR = DATA_DIR + r"\EDDIES"
FIG_DIR  = DATA_DIR + r"\FIGURES\PUBLICATION"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Chargement ────────────────────────────────────────────────────────────
print("Chargement catalogues eddies...")
df_a = pd.read_csv(os.path.join(EDDY_DIR, "anticyclones_1993_2023_ALL.csv"))
df_c = pd.read_csv(os.path.join(EDDY_DIR, "cyclones_1993_2023_ALL.csv"))

for df in [df_a, df_c]:
    df['date']  = pd.to_datetime(df['date'])
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month

MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
months    = np.arange(1, 13)
years_all = np.arange(1993, 2024)
n_years   = 31

na_yr = [len(df_a[df_a['year']==y]) for y in years_all]
nc_yr = [len(df_c[df_c['year']==y]) for y in years_all]
na_m  = [len(df_a[df_a['month']==m])/n_years for m in months]
nc_m  = [len(df_c[df_c['month']==m])/n_years for m in months]

coeffs_a   = np.polyfit(years_all, na_yr, 1)
trend_a    = np.polyval(coeffs_a, years_all)
slope_a_yr = coeffs_a[0] * 10

print(f"Anticyclones : {len(df_a):,} | Cyclones : {len(df_c):,}")

# ── Configuration ─────────────────────────────────────────────────────────
proj       = ccrs.PlateCarree()
EXTENT     = [-5, 10, 35, 41]
LAND_COLOR = '#DDD5C0'
OCEAN_BG   = '#D6E4F0'    # ← bleu plus visible pour contraster avec hexbin
LABEL_KW   = dict(
    fontsize=9, fontweight='bold', va='top', ha='left',
    color='black', zorder=20,
    bbox=dict(facecolor='white', alpha=0.88,
              edgecolor='none', boxstyle='round,pad=0.2')
)
GRID_KW = dict(alpha=0.35, linestyle='--', linewidth=0.35, color='#888888')

fig = plt.figure(figsize=(7.0, 7.0))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.06,
    wspace=0.40,
    left=0.05,
    right=0.91,
    top=0.97,
    bottom=0.07,
    height_ratios=[1.05, 1.05, 0.75]
)

# ── Fonction setup ─────────────────────────────────────────────────────────
def setup_geoax(ax, label, title):
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
    gl.xlabel_style = {'size': 5.8, 'color': '#333333'}
    gl.ylabel_style = {'size': 5.8, 'color': '#333333'}
    ax.text(0.014, 0.958, label, transform=ax.transAxes, **LABEL_KW)
    ax.set_title(title, fontsize=7.0, pad=2.5, loc='center', color='#111111')

cbar_items = []

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (a) — Densité anticyclones
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
setup_geoax(ax1, '(a)', f'Anticyclone density  (N = {len(df_a):,})')

hb1 = ax1.hexbin(
    df_a['lon'], df_a['lat'],
    gridsize=20,
    cmap=cmocean.cm.matter,
    mincnt=1,
    alpha=1.0,
    transform=proj,
    zorder=3,
    linewidths=0.0,
    bins='log'
)
cbar_items.append((hb1, ax1, 'Detections (log scale)', None, None))

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (b) — Densité cyclones
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
setup_geoax(ax2, '(b)', f'Cyclone density  (N = {len(df_c):,})')

hb2 = ax2.hexbin(
    df_c['lon'], df_c['lat'],
    gridsize=20,
    cmap=cmocean.cm.deep,
    mincnt=1,
    alpha=1.0,
    transform=proj,
    zorder=3,
    linewidths=0.0,
    bins='log'
)
cbar_items.append((hb2, ax2, 'Detections (log scale)', None, None))

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (c) — Amplitude anticyclones
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0], projection=proj)
setup_geoax(ax3, '(c)', 'Anticyclone amplitude (cm)')

hb3 = ax3.hexbin(
    df_a['lon'], df_a['lat'],
    C=df_a['amplitude']*100,
    gridsize=20,
    cmap=cmocean.cm.thermal,
    reduce_C_function=np.mean,
    mincnt=1,
    vmin=0, vmax=10,
    alpha=1.0,
    transform=proj, zorder=3,
    linewidths=0.0
)
cbar_items.append((hb3, ax3, 'Mean amplitude (cm)', '%.0f', [0,2,4,6,8,10]))


# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (d) — Rayon anticyclones
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1], projection=proj)
setup_geoax(ax4, '(d)', 'Anticyclone speed radius (km)')

hb4 = ax4.hexbin(
    df_a['lon'], df_a['lat'],
    C=df_a['radius_s']/1000,
    gridsize=20,
    cmap=cmocean.cm.speed,
    reduce_C_function=np.mean,
    mincnt=1,
    vmin=10, vmax=45,
    alpha=1.0,
    transform=proj, zorder=3,
    linewidths=0.0
)
cbar_items.append((hb4, ax4, 'Mean radius (km)', '%.0f', [10,20,30,40]))

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (e) — Série temporelle annuelle
# ═══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, 0])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_linewidth(0.5)
ax5.spines['bottom'].set_linewidth(0.5)
ax5.grid(True, **GRID_KW)

w = 0.38
ax5.bar(years_all - w/2, na_yr, w,
        label=f'Anticyclones  (mean = {int(np.mean(na_yr))}/yr)',
        color='#C0392B', alpha=0.78,
        edgecolor='#7B241C', linewidth=0.4)
ax5.bar(years_all + w/2, nc_yr, w,
        label=f'Cyclones  (mean = {int(np.mean(nc_yr))}/yr)',
        color='#2166AC', alpha=0.78,
        edgecolor='#1A3A5C', linewidth=0.4)
ax5.plot(years_all, trend_a,
         color='#C0392B', linewidth=1.3, linestyle='--',
         label=f'Anti. trend  ({slope_a_yr:+.0f} decade$^{{-1}}$)',
         zorder=5)

# Annotation 2003 — flèche vers le HAUT
idx_2003 = int(np.where(years_all == 2003)[0][0])
y_bar    = na_yr[idx_2003] 
ax5.annotate(
    '2003*',
    xy=(2003, y_bar),           # ← pointe juste au-dessus de la barre
    xytext=(2006, y_bar + 60),    # ← texte encore plus haut et décalé
    fontsize=5.5, color='#7D3C98',
    ha='center', fontweight='bold',
    arrowprops=dict(
        arrowstyle='->',
        color='#7D3C98',
        lw=0.8,
        connectionstyle='arc3,rad=-0.2'
    )
)

ax5.set_xlim(1992, 2024)
ax5.set_xlabel('Year', fontsize=7)
ax5.set_ylabel('Annual detections', fontsize=7)
ax5.set_title('(e)  Annual eddy detections (1993-2023)',
              fontsize=8, fontweight='bold', loc='left', pad=3)
leg5 = ax5.legend(framealpha=0.88, edgecolor='#CCCCCC',
                  fancybox=False, fontsize=5.8,
                  handlelength=1.5, borderpad=0.4,
                  loc='upper left', labelspacing=0.25)
leg5.get_frame().set_linewidth(0.4)
ax5.tick_params(axis='x', labelsize=6.0, rotation=45)

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (f) — Cycle saisonnier
# ═══════════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[2, 1])
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_linewidth(0.5)
ax6.spines['bottom'].set_linewidth(0.5)
ax6.grid(True, **GRID_KW)

x  = np.arange(12)
w2 = 0.38
bars_a = ax6.bar(x - w2/2, na_m, w2,
                 label='Anticyclones',
                 color='#C0392B', alpha=0.78,
                 edgecolor='#7B241C', linewidth=0.4)
bars_c = ax6.bar(x + w2/2, nc_m, w2,
                 label='Cyclones',
                 color='#2166AC', alpha=0.78,
                 edgecolor='#1A3A5C', linewidth=0.4)

max_a_m = int(np.argmax(na_m))
max_c_m = int(np.argmax(nc_m))
bars_a[max_a_m].set_linewidth(1.6)
bars_c[max_c_m].set_linewidth(1.6)

ax6.annotate(
    f'Max: {MONTH_NAMES[max_a_m]}',
    xy=(max_a_m - w2/2, na_m[max_a_m]),
    xytext=(max_a_m + 1.2, na_m[max_a_m] * 1.04),
    fontsize=5.5, color='#C0392B', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.7)
)
ax6.annotate(
    f'Max: {MONTH_NAMES[max_c_m]}',
    xy=(max_c_m + w2/2, nc_m[max_c_m]),
    xytext=(max_c_m - 2.8, nc_m[max_c_m] * 1.08),
    fontsize=5.5, color='#2166AC', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#2166AC', lw=0.7)
)

ax6.set_xticks(x)
ax6.set_xticklabels(MONTH_NAMES, fontsize=6.0)
ax6.set_xlabel('Month', fontsize=7)
ax6.set_ylabel('Mean detections yr$^{-1}$', fontsize=7)
ax6.set_title('(f)  Seasonal cycle of eddy detections',
              fontsize=8, fontweight='bold', loc='left', pad=3)
leg6 = ax6.legend(framealpha=0.88, edgecolor='#CCCCCC',
                  fancybox=False, fontsize=6.0,
                  handlelength=1.2, borderpad=0.4,
                  loc='upper left', labelspacing=0.25)
leg6.get_frame().set_linewidth(0.4)

# ── Colorbars ─────────────────────────────────────────────────────────────
fig.canvas.draw()
for idx, (mappable, ax, label, fmt, ticks) in enumerate(cbar_items):
    pos = ax.get_position()
    cax = fig.add_axes([
        pos.x1 + 0.009,
        pos.y0 + 0.010,
        0.012,
        pos.height - 0.020
    ])
    cb = plt.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=5.8, labelpad=2)
    cb.ax.tick_params(labelsize=5.2, width=0.4, length=2, pad=1)
    cb.outline.set_linewidth(0.4)

    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
    elif idx in [0, 1]:
        # hexbin log scale — ticks manuels propres
        cb.set_ticks([1, 10, 100, 1000])
        cb.set_ticklabels(['1', '10', '100', '1000'])
        cb.set_label(label.replace('log scale', 'log₁₀'), fontsize=5.8, labelpad=2)
    else:
        cb.locator = mticker.MaxNLocator(nbins=5, prune='both')
        cb.update_ticks()



# ── Sauvegarde ────────────────────────────────────────────────────────────
OUTPUT_PNG = FIG_DIR + r"\Figure_06_Eddy_Atlas.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_06_Eddy_Atlas.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()
print("Figure 6 terminee.")
