# Figure 6 - dynamic zonation (Ward clustering) and surface divergence
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
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

# ── Chargement + calculs ──────────────────────────────────────────────────
print("Chargement MEDSEA PHY courants...")
ds_cur = xr.open_dataset(
    DATA_DIR + r"\cmems_mod_med_phy-cur_my_4.2km_P1D-m_uo-vo"
               r"_5.00W-10.00E_35.02N-40.98N_1.02m_1993-01-01-2023-12-31.nc"
)
u_mean = ds_cur['uo'].mean('time').squeeze().values
v_mean = ds_cur['vo'].mean('time').squeeze().values
lats   = ds_cur['latitude'].values
lons   = ds_cur['longitude'].values
ds_cur.close()

# ── Divergence ────────────────────────────────────────────────────────────
dlat_m = np.abs(np.diff(lats).mean()) * 111000
dlon_m = np.abs(np.diff(lons).mean()) * 111000 * \
         np.cos(np.radians(np.mean(lats)))
u_fg   = np.where(np.isnan(u_mean), 0.0, u_mean)
v_fg   = np.where(np.isnan(v_mean), 0.0, v_mean)
div    = np.gradient(u_fg, dlon_m, axis=1) + \
         np.gradient(v_fg, dlat_m, axis=0)
div    = np.where(np.isnan(u_mean)|np.isnan(v_mean), np.nan, div)

# ── CAH ───────────────────────────────────────────────────────────────────
print("CAH Ward...")
nlat, nlon   = u_mean.shape
mask_valid   = ~(np.isnan(u_mean)|np.isnan(v_mean))
u_flat       = u_mean[mask_valid]
v_flat       = v_mean[mask_valid]
div_flat     = div[mask_valid]
X_norm       = StandardScaler().fit_transform(
                   np.column_stack([u_flat, v_flat]))
Z            = linkage(X_norm, method='ward')
ch_scores    = {}
for n in range(3, 7):
    lbl          = fcluster(Z, t=n, criterion='maxclust')
    ch_scores[n] = calinski_harabasz_score(X_norm, lbl)
n_opt        = max(ch_scores, key=ch_scores.get)
labels_flat  = fcluster(Z, t=n_opt, criterion='maxclust')
labels_2d    = np.full((nlat, nlon), np.nan)
labels_2d[mask_valid] = labels_flat

cluster_stats = {}
for c in range(1, n_opt+1):
    m   = labels_flat == c
    u_c = u_flat[m]; v_c = v_flat[m]
    spd = np.sqrt(u_c**2 + v_c**2)
    dv  = div_flat[m]
    cluster_stats[c] = {
        'n':   m.sum(),
        'u':   np.nanmean(u_c)*100,
        'v':   np.nanmean(v_c)*100,
        'spd': np.nanmean(spd)*100,
        'div': np.nanmean(dv)*1e6,
    }
print(f"N optimal = {n_opt}")

# ── Palette zones — couleurs distinctes qualitatives premium ──────────────
# Inspiré des palettes ColorBrewer Set1 — lisibles en niveaux de gris aussi
ZONE_COLORS = [
    '#2166AC',   # bleu profond     Zone 1
    '#D6604D',   # rouge brique     Zone 2
    '#4DAC26',   # vert forêt       Zone 3
    '#762A83',   # violet           Zone 4
    '#E08214',   # orange foncé     Zone 5
    '#35978F',   # teal             Zone 6
][:n_opt]

# ── Figure ────────────────────────────────────────────────────────────────
proj       = ccrs.PlateCarree()
EXTENT     = [-5, 10, 35, 41]
LAND_COLOR = '#DDD5C0'
OCEAN_BG   = '#EEF2F6'
LABEL_KW   = dict(fontsize=9, fontweight='bold', va='top', ha='left',
                  color='black', zorder=10,
                  bbox=dict(facecolor='white', alpha=0.85,
                            edgecolor='none', boxstyle='round,pad=0.2'))
GRID_KW    = dict(alpha=0.35, linestyle='--',
                  linewidth=0.35, color='#888888')

# Hauteur calculée : 2 rangées, ratio [carte, graphes] = [1.15, 0.85]
fig = plt.figure(figsize=(7.0, 4.8))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.32,
    wspace=0.42,
    left=0.05,
    right=0.91,
    top=0.96,
    bottom=0.09,
    height_ratios=[1.0, 0.82]
)

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
    ax.set_title(title, fontsize=7.0, pad=2.5, loc='center',
                 color='#111111')

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (a) — Zonation CAH
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
setup_geoax(ax1, '(a)',
            f'Dynamic circulation regimes — Ward HCA (N = {n_opt})')

cmap_cah = mcolors.ListedColormap(ZONE_COLORS)
bounds   = np.arange(0.5, n_opt+1.5, 1)
norm_cah = mcolors.BoundaryNorm(bounds, cmap_cah.N)

ax1.pcolormesh(lons, lats, labels_2d,
               cmap=cmap_cah, norm=norm_cah,
               transform=proj, zorder=2, rasterized=True)

# Légende compacte dans le coin — sans la colorbar (redondant)
# On utilise des patches colorés discrets
import matplotlib.patches as mpatches
patches = []
for c in range(1, n_opt+1):
    dv   = cluster_stats[c]['div']
    spd  = cluster_stats[c]['spd']
    icon = '↑' if dv > 0 else '↓'
    patches.append(
        mpatches.Patch(
            facecolor=ZONE_COLORS[c-1],
            edgecolor='white',
            linewidth=0.5,
            label=f'Z{c}  {spd:.1f} cm s$^{{-1}}$  {icon}'
        )
    )

leg1 = ax1.legend(
    handles=patches,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    framealpha=0.92,
    edgecolor='#CCCCCC',
    fancybox=False,
    fontsize=5.6,
    handlelength=1.0,
    handleheight=0.85,
    borderpad=0.45,
    labelspacing=0.18,
    title='Zone   Speed        Regime',
    title_fontsize=5.5,
    ncol=1
)
leg1.get_frame().set_linewidth(0.4)

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (b) — Divergence
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
setup_geoax(ax2, '(b)',
            'Surface divergence $\\nabla \\cdot \\mathbf{u}$  '
            '($\\times 10^{-6}$ s$^{-1}$)')

div_plot = np.where(mask_valid, div, np.nan)
vmax_div = np.nanpercentile(np.abs(div_plot[~np.isnan(div_plot)]), 95)

cf2 = ax2.contourf(
    lons, lats, div_plot*1e6,
    levels=np.linspace(-vmax_div*1e6, vmax_div*1e6, 20),
    cmap=cmocean.cm.curl,
    transform=proj, zorder=2, extend='both'
)
# Une seule ligne de zéro — propre
ax2.contour(lons, lats, div_plot*1e6,
            levels=[0],
            colors='#222222', linewidths=0.7,
            transform=proj, zorder=3,
            linestyles='-')

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (c) — Calinski-Harabasz
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['bottom'].set_linewidth(0.5)
ax3.grid(True, **GRID_KW)

ns   = list(ch_scores.keys())
chs  = list(ch_scores.values())
# Couleurs : bleu pâle pour tous, rouge pour optimal
bar_colors = ['#90CAF9'] * len(ns)
bar_colors[n_opt - min(ns)] = '#C0392B'
bars = ax3.bar(ns, chs,
               color=bar_colors,
               edgecolor=['#1A5F7A' if c != '#C0392B' else '#7B241C'
                          for c in bar_colors],
               linewidth=0.6, alpha=0.90, width=0.55, zorder=3)

ax3.set_xticks(ns)
ax3.set_xticklabels([str(n) for n in ns], fontsize=7)
ax3.set_xlabel('Number of clusters', fontsize=7)
ax3.set_ylabel('Calinski-Harabasz index', fontsize=7)
ax3.set_title('(c)  Calinski-Harabasz criterion',
              fontsize=7.5, fontweight='bold', loc='left', pad=3)

# Annotation de la valeur optimale
ax3.annotate(f'Optimal\nN = {n_opt}',
             xy=(n_opt, ch_scores[n_opt]),
             xytext=(n_opt+0.25, ch_scores[n_opt]*0.94),
             fontsize=6.0, color='#C0392B', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.7))

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (d) — Profils de vitesse
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(0.5)
ax4.spines['bottom'].set_linewidth(0.5)
ax4.grid(True, **GRID_KW)

clusters = list(cluster_stats.keys())
x        = np.arange(len(clusters))
w        = 0.25
spds     = [cluster_stats[c]['spd'] for c in clusters]
u_v      = [cluster_stats[c]['u']   for c in clusters]
v_v      = [cluster_stats[c]['v']   for c in clusters]
divs     = [cluster_stats[c]['div'] for c in clusters]

b1 = ax4.bar(x-w,  spds, w, label='Total speed',
             color='#5BA4CF', edgecolor='#1A5F7A', lw=0.5, alpha=0.88)
b2 = ax4.bar(x,    u_v,  w, label='U (zonal)',
             color='#4CAF50', edgecolor='#1B5E20', lw=0.5, alpha=0.88)
b3 = ax4.bar(x+w,  v_v,  w, label='V (meridional)',
             color='#FF9800', edgecolor='#E65100', lw=0.5, alpha=0.88)

ax4.axhline(0, color='#333333', linewidth=0.6, zorder=5)

# Couleur de fond différente par zone (bande verticale discrète)
for i, c in enumerate(clusters):
    ax4.axvspan(i-0.48, i+0.48,
                color=ZONE_COLORS[c-1], alpha=0.07, zorder=0)

ax4.set_xticks(x)
ax4.set_xticklabels([f'Z{c}' for c in clusters], fontsize=6.5)
ax4.set_ylabel('Velocity (cm s$^{-1}$)', fontsize=7)
ax4.set_xlabel('Dynamic zone', fontsize=7)
ax4.set_title('(d)  Velocity profiles per zone',
              fontsize=7.5, fontweight='bold', loc='left', pad=3)

# Annotations UP/DW — à l'intérieur du cadre, compactes
ax4.set_ylim(bottom=None, top=max(spds)*1.35)   # ← forcer ylim avant annotation
y_annot = max(spds) * 1.20
for i, c in enumerate(clusters):
    dv  = divs[i]
    txt = '↑' if dv > 0 else '↓'
    col = '#C0392B' if dv > 0 else '#2166AC'
    ax4.text(i, y_annot, txt, ha='center',
             fontsize=7, color=col, fontweight='bold')

leg4 = ax4.legend(
    framealpha=0.88, edgecolor='#CCCCCC',
    fancybox=False, fontsize=6.0,
    handlelength=1.2, borderpad=0.4,
    loc='lower right'
)
leg4.get_frame().set_linewidth(0.4)

# ── Colorbars après rendu ─────────────────────────────────────────────────
fig.canvas.draw()

# (a) — Colorbar discrète zones (verticale, compacte)
pos1 = ax1.get_position()
# (b) — Colorbar divergence
pos2 = ax2.get_position()
cax2 = fig.add_axes([pos2.x1+0.009, pos2.y0+0.010,
                     0.012, pos2.height-0.020])
cb2  = plt.colorbar(cf2, cax=cax2, format='%.2f')
cb2.set_label('Divergence ($\\times 10^{-6}$ s$^{-1}$)',
              fontsize=5.5, labelpad=2)
cb2.ax.tick_params(labelsize=5.0, width=0.4, length=2, pad=1)
cb2.outline.set_linewidth(0.4)
cb2.locator = mticker.MaxNLocator(nbins=5, prune='both')
cb2.update_ticks()

# ── Sauvegarde ────────────────────────────────────────────────────────────
OUTPUT_PNG = FIG_DIR + r"\Figure_05_CAH_zonation.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_05_CAH_zonation.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()
print("Figure 5 terminee.")
