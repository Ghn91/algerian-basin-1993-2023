"""
Figure 2 - mean ADT, EKE, SLA std and MDT (ALL-SAT).

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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cmocean
import os, warnings
warnings.filterwarnings('ignore')

# ── Paramètres globaux publication ────────────────────────────────────────
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
})

FIG_DIR  = DATA_DIR + r"\FIGURES\PUBLICATION"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Chargement ────────────────────────────────────────────────────────────
print("Chargement des donnees...")
ds_alti = xr.open_dataset(
    DATA_DIR + r"\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg"
               r"_P1D_multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)
ds_mdt = xr.open_dataset(
    DATA_DIR + r"\cmems_obs-sl_eur_phy-mdt_my_l4-0.0625deg_P20Y"
               r"_mdt-v-u_4.97W-9.97E_35.03N-40.97N_2003-01-01.nc"
)

# ── Calculs ───────────────────────────────────────────────────────────────
print("Calculs en cours...")
adt_mean  = ds_alti['adt'].mean('time')
sla_std   = ds_alti['sla'].std('time') * 100
eke_mean  = (0.5*(ds_alti['ugosa']**2 +
                  ds_alti['vgosa']**2) * 1e4).mean('time')
ugos_mean = ds_alti['ugos'].mean('time')
vgos_mean = ds_alti['vgos'].mean('time')
mdt_2d    = ds_mdt['mdt'].squeeze()
mdt_u     = ds_mdt['u'].squeeze()
mdt_v     = ds_mdt['v'].squeeze()
print("Calculs termines.")

# ── Constantes mise en page ───────────────────────────────────────────────
proj       = ccrs.PlateCarree()
EXTENT     = [-5, 10, 35, 41]
LAND_COLOR = '#DDD5C0'
OCEAN_BG   = '#EEF2F6'
LABEL_KW   = dict(fontsize=9, fontweight='bold', va='top', ha='left',
                  color='black', zorder=10,
                  bbox=dict(facecolor='white', alpha=0.80,
                            edgecolor='none', boxstyle='round,pad=0.2'))

# ── Figure — dimensions double colonne ────────────────────────────────────
# Largeur 178 mm = 7.0 inches
# On laisse matplotlib gérer la hauteur via l'aspect cartopy
fig = plt.figure(figsize=(7.0, 3.9))
fig.patch.set_facecolor('white')

# GridSpec — marges serrées, espace entre colonnes pour les colorbars
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    hspace=0.18,
    wspace=0.40,
    left=0.05,
    right=0.91,
    top=0.96,
    bottom=0.09
)

# ─────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────
def setup_ax(ax, label, title):
    """Configure un GeoAxes au standard publication."""
    ax.set_extent(EXTENT, crs=proj)
    ax.set_facecolor(OCEAN_BG)
    ax.add_feature(cfeature.LAND,
                   facecolor=LAND_COLOR, zorder=5, linewidth=0)
    ax.add_feature(cfeature.COASTLINE,
                   linewidth=0.45, color='#444444', zorder=6)
    ax.add_feature(cfeature.BORDERS,
                   linewidth=0.3, linestyle=':', color='#888888', zorder=6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.22,
                      color='#AAAAAA', alpha=0.7, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlocator     = mticker.FixedLocator([-4, -2, 0, 2, 4, 6, 8])
    gl.ylocator     = mticker.FixedLocator([36, 37, 38, 39, 40])
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 6.0, 'color': '#333333'}
    gl.ylabel_style = {'size': 6.0, 'color': '#333333'}
    # Label panneau dans le coin supérieur gauche
    ax.text(0.014, 0.962, label, transform=ax.transAxes, **LABEL_KW)
    # Titre centré au-dessus
    ax.set_title(title, fontsize=7.0, pad=2.5, loc='center',
                 color='#1A1A1A')


def add_cbar(fig, mappable, ax, label, fmt, ticks=None):
    """
    Colorbar verticale positionnée manuellement
    à droite de l'axe — aucun chevauchement garanti.
    """
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([
        pos.x1 + 0.010,       # décalage à droite de l'axe
        pos.y0 + 0.015,       # marge basse
        0.013,                # largeur colorbar
        pos.height - 0.030    # hauteur
    ])
    cb = fig.colorbar(mappable, cax=cax, format=fmt)
    cb.set_label(label, fontsize=6.5, labelpad=3)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(int(t)) if t == int(t) else str(t)
                           for t in ticks])
    else:
        cb.locator = mticker.MaxNLocator(nbins=5, prune='both')
        cb.update_ticks()
    return cb


# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (a) — ADT + VECTEURS COURANT
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0], projection=proj)
setup_ax(ax1, '(a)', 'Mean ADT and geostrophic currents (1993-2023)')

cf1 = ax1.contourf(
    adt_mean.longitude, adt_mean.latitude, adt_mean.values,
    levels=np.linspace(-0.14, 0.23, 28),
    cmap=cmocean.cm.deep, transform=proj, zorder=2, extend='both'
)
ax1.contour(
    adt_mean.longitude, adt_mean.latitude, adt_mean.values,
    levels=np.arange(-0.12, 0.22, 0.03),
    colors='white', linewidths=0.20, alpha=0.45,
    transform=proj, zorder=3
)
# Vecteurs courant
step = 5
U_v  = ugos_mean.values[::step, ::step].copy()
V_v  = vgos_mean.values[::step, ::step].copy()
spd  = np.sqrt(U_v**2 + V_v**2)
U_v[spd < 0.02] = np.nan
V_v[spd < 0.02] = np.nan
q1 = ax1.quiver(
    ugos_mean.longitude.values[::step],
    ugos_mean.latitude.values[::step],
    U_v, V_v,
    transform=proj, zorder=4,
    scale=4.5, width=0.0025,
    headwidth=3.0, headlength=3.5,
    color='white', alpha=0.90
)
ax1.quiverkey(q1, X=0.76, Y=0.06, U=0.3,
              label='0.3 m/s', labelpos='E',
              fontproperties={'size': 5.5},
              color='white', labelcolor='white',
              coordinates='axes')

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (b) — EKE MOYENNE
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1], projection=proj)
setup_ax(ax2, '(b)', 'Mean EKE (1993-2023)')

eke_max = float(eke_mean.quantile(0.98))
cf2 = ax2.contourf(
    eke_mean.longitude, eke_mean.latitude, eke_mean.values,
    levels=np.linspace(0, eke_max, 22),
    cmap=cmocean.cm.matter, transform=proj, zorder=2, extend='max'
)
ax2.contour(
    adt_mean.longitude, adt_mean.latitude, adt_mean.values,
    levels=np.arange(-0.10, 0.20, 0.05),
    colors='#1A1A1A', linewidths=0.22, alpha=0.30,
    transform=proj, zorder=3
)

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (c) — SLA STD
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0], projection=proj)
setup_ax(ax3, '(c)', 'SLA standard deviation (1993-2023)')

cf3 = ax3.contourf(
    sla_std.longitude, sla_std.latitude, sla_std.values,
    levels=np.linspace(4.5, 12.0, 22),
    cmap=cmocean.cm.amp, transform=proj, zorder=2, extend='both'
)
ax3.contour(
    sla_std.longitude, sla_std.latitude, sla_std.values,
    levels=[8.0, 10.0],
    colors=['#333333', '#111111'],
    linewidths=[0.45, 0.75],
    linestyles=['--', '-'],
    transform=proj, zorder=3
)

# ═══════════════════════════════════════════════════════════════════════════
# PANNEAU (d) — MDT + VECTEURS
# ═══════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1], projection=proj)
setup_ax(ax4, '(d)', 'Mean Dynamic Topography (MDT-CMEMS_2024_EUR)')

cf4 = ax4.contourf(
    mdt_2d.longitude, mdt_2d.latitude, mdt_2d.values,
    levels=np.linspace(-0.12, 0.23, 28),
    cmap=cmocean.cm.curl, transform=proj, zorder=2, extend='both'
)
ax4.contour(
    mdt_2d.longitude, mdt_2d.latitude, mdt_2d.values,
    levels=np.arange(-0.09, 0.22, 0.04),
    colors='black', linewidths=0.20, alpha=0.30,
    transform=proj, zorder=3
)
step_m = 7   # vecteurs bien espacés
q4 = ax4.quiver(
    mdt_2d.longitude.values[::step_m],
    mdt_2d.latitude.values[::step_m],
    mdt_u.values[::step_m, ::step_m],
    mdt_v.values[::step_m, ::step_m],
    transform=proj, zorder=4,
    scale=4.0, width=0.003,
    headwidth=3.0, headlength=3.5,
    color='black', alpha=0.45
)

# ═══════════════════════════════════════════════════════════════════════════
# COLORBARS — positionnement manuel après rendu
# ═══════════════════════════════════════════════════════════════════════════
fig.canvas.draw()

add_cbar(fig, cf1, ax1, 'ADT (m)',
         fmt='%.2f')

add_cbar(fig, cf2, ax2, 'EKE (cm$^2$ s$^{-2}$)',
         fmt='%d',
         ticks=[0, 50, 100, 150, 200])

add_cbar(fig, cf3, ax3, 'SLA std (cm)',
         fmt='%.1f')

add_cbar(fig, cf4, ax4, 'MDT (m)',
         fmt='%.2f')

# ═══════════════════════════════════════════════════════════════════════════
# SAUVEGARDE
# ═══════════════════════════════════════════════════════════════════════════
OUTPUT_PNG = FIG_DIR + r"\Figure_01_mean_circulation.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_01_mean_circulation.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()

ds_alti.close()
ds_mdt.close()
print("Figure 1 terminee.")
