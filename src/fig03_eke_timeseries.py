"""
Figure 3 - basin-averaged EKE time series, seasonal cycle, anomaly (ALL-SAT).

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
import pymannkendall as mk
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':     'DejaVu Sans',
    'font.size':        8,
    'axes.labelsize':   8,
    'axes.titlesize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'savefig.dpi':     300,
    'lines.linewidth':  0.8,
    'axes.linewidth':   0.5,
    'xtick.major.width':0.5,
    'ytick.major.width':0.5,
})

FIG_DIR  = DATA_DIR + r"\FIGURES\PUBLICATION"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Chargement ALL-SAT ────────────────────────────────────────────────────
print("Chargement ALL-SAT...")
ds = xr.open_dataset(
    DATA_DIR + r"\cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg"
               r"_P1D_multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc"
)

# ── Calcul EKE ALL-SAT ────────────────────────────────────────────────────
print("Calcul EKE ALL-SAT...")
eke_daily  = (0.5*(ds['ugosa']**2 + ds['vgosa']**2)*1e4
              ).mean(dim=['latitude','longitude'], skipna=True)
eke_mon    = eke_daily.resample(time='ME').mean()
eke_clim   = eke_mon.groupby('time.month').mean()
eke_anom   = eke_mon.groupby('time.month') - eke_clim
eke_smooth = uniform_filter1d(eke_mon.values, size=12, mode='nearest')
times_m    = eke_mon.time.values
N          = len(times_m)

# Tendance linéaire ALL-SAT
t_num   = np.arange(N)
coeffs  = np.polyfit(t_num, eke_mon.values, 1)
trend_line = np.polyval(coeffs, t_num)
slope_all  = coeffs[0] * 12 * 10   # cm²/s²/décennie

# Climatologie saisonnière
eke_seas  = eke_clim.values
eke_std_m = eke_mon.groupby('time.month').std().values
months    = np.arange(1, 13)
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

# Anomalie désaisonnalisée lissée
anom_smooth = uniform_filter1d(eke_anom.values, size=12, mode='nearest')

ds.close()
print("Calculs termines.")

# ── Figure 2 ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 6.8))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 1,
    figure=fig,
    hspace=0.38,
    left=0.09, right=0.97,
    top=0.96,  bottom=0.07
)

# Couleurs cohérentes
C_FILL  = '#AED6E8'   # bleu pâle fill
C_LINE  = '#1A5F7A'   # bleu foncé ligne
C_TREND = '#C0392B'   # rouge tendance
C_POS   = '#E8A09A'   # rouge pâle anomalie positive
C_NEG   = '#9ABFDD'   # bleu pâle anomalie négative
C_SMTH  = '#1A1A1A'   # noir lissage

GRID_KW = dict(alpha=0.35, linestyle='--', linewidth=0.35, color='#888888')

def style_ax(ax, xlabel=False):
    ax.grid(True, **GRID_KW)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    if not xlabel:
        ax.tick_params(labelbottom=False)

# ─────────────────────────────────────────────────────────────────────────
# PANNEAU (a) — Série temporelle EKE + tendance
# ─────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1)

ax1.fill_between(times_m, eke_mon.values,
                 alpha=0.30, color=C_FILL, linewidth=0)
ax1.plot(times_m, eke_mon.values,
         color=C_LINE, linewidth=0.5, alpha=0.75,
         label='Monthly EKE')
ax1.plot(times_m, eke_smooth,
         color=C_LINE, linewidth=1.8,
         label='12-month running mean')
ax1.plot(times_m, trend_line,
         color=C_TREND, linewidth=1.5, linestyle='--',
         label=f'Linear trend ({slope_all:+.2f} cm$^2$ s$^{{-2}}$ decade$^{{-1}}$)')

# Annotation statistiques dans le panneau
stat_txt = (f'Mean = {float(eke_mon.mean()):.1f} cm$^2$ s$^{{-2}}$\n'
            f'Std  = {float(eke_mon.std()):.1f}  cm$^2$ s$^{{-2}}$\n'
            f'Max  = {float(eke_mon.max()):.1f}  cm$^2$ s$^{{-2}}$')
ax1.text(0.012, 0.97, stat_txt,
         transform=ax1.transAxes,
         fontsize=6.0, va='top', ha='left',
         bbox=dict(facecolor='white', alpha=0.80,
                   edgecolor='#CCCCCC', linewidth=0.4,
                   boxstyle='round,pad=0.35'))

ax1.set_xlim(times_m[0], times_m[-1])
ax1.set_ylabel('EKE (cm$^2$ s$^{-2}$)', fontsize=7.5)
ax1.set_title('(a)  Basin-averaged EKE time series (ALL-SAT, 1993-2023)',
              fontsize=8, fontweight='bold', loc='left', pad=3)

leg1 = ax1.legend(loc='upper left', framealpha=0.85,
                  edgecolor='#CCCCCC', fancybox=False,
                  handlelength=1.8, handletextpad=0.5,
                  borderpad=0.5, labelspacing=0.3)
leg1.get_frame().set_linewidth(0.4)

ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ─────────────────────────────────────────────────────────────────────────
# PANNEAU (b) — Cycle saisonnier
# ─────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2)

bars = ax2.bar(months, eke_seas,
               color=C_FILL, edgecolor=C_LINE,
               linewidth=0.6, alpha=0.85,
               width=0.65, zorder=3,
               label='Monthly climatology')
ax2.errorbar(months, eke_seas, yerr=eke_std_m,
             fmt='none', color=C_LINE,
             capsize=3.5, capthick=0.7,
             linewidth=0.7, zorder=4)

# Surligner max et min
max_idx = int(np.argmax(eke_seas))
min_idx = int(np.argmin(eke_seas))
bars[max_idx].set_facecolor('#E84040')
bars[max_idx].set_edgecolor('#8B0000')
bars[min_idx].set_facecolor('#6EB5D6')
bars[min_idx].set_edgecolor('#1A5F7A')

ax2.annotate(f'Max: {MONTH_NAMES[max_idx]}',
             xy=(months[max_idx], eke_seas[max_idx] + eke_std_m[max_idx]),
             xytext=(months[max_idx] + 0.6,
                     eke_seas[max_idx] + eke_std_m[max_idx] + 1.5),
             fontsize=6.5, color='#8B0000', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#8B0000',
                             lw=0.8))
ax2.annotate(f'Min: {MONTH_NAMES[min_idx]}',
             xy=(months[min_idx], eke_seas[min_idx] - eke_std_m[min_idx]),
             xytext=(months[min_idx] + 0.6,
                     eke_seas[min_idx] - eke_std_m[min_idx] - 4),
             fontsize=6.5, color='#1A5F7A', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1A5F7A',
                             lw=0.8))

ax2.set_xticks(months)
ax2.set_xticklabels(MONTH_NAMES, fontsize=6.5)
ax2.set_ylabel('EKE (cm$^2$ s$^{-2}$)', fontsize=7.5)
ax2.set_title('(b)  Seasonal cycle of EKE — climatology 1993-2023',
              fontsize=8, fontweight='bold', loc='left', pad=3)
ax2.legend(framealpha=0.85, edgecolor='#CCCCCC',
           fancybox=False, handlelength=1.2,
           borderpad=0.5).get_frame().set_linewidth(0.4)

# ─────────────────────────────────────────────────────────────────────────
# PANNEAU (c) — Anomalies interannuelles désaisonnalisées
# ─────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, xlabel=True)

anom_vals = eke_anom.values
ax3.fill_between(times_m, anom_vals,
                 where=anom_vals >= 0,
                 color=C_POS, alpha=0.65,
                 linewidth=0, label='Positive anomaly')
ax3.fill_between(times_m, anom_vals,
                 where=anom_vals < 0,
                 color=C_NEG, alpha=0.65,
                 linewidth=0, label='Negative anomaly')
ax3.plot(times_m, anom_smooth,
         color=C_SMTH, linewidth=1.6,
         label='12-month running mean')
ax3.axhline(0, color='black', linewidth=0.7, zorder=5)

# Ligne verticale rupture de regime ~2008
import matplotlib.patches as mpatches
ax3.axvline(np.datetime64('2008-01-01'), color='#555555',
            linewidth=1.0, linestyle=':', zorder=4,
            label='Regime shift (~2008)')
ax3.text(np.datetime64('2008-04-01'),
         ax3.get_ylim()[0] if ax3.get_ylim()[0] != 0 else -25,
         'Regime\nshift', fontsize=5.5, color='#555555',
         va='bottom', ha='left')

ax3.set_xlim(times_m[0], times_m[-1])
ax3.set_ylabel('EKE anomaly (cm$^2$ s$^{-2}$)', fontsize=7.5)
ax3.set_xlabel('Year', fontsize=7.5)
ax3.set_title('(c)  Interannual EKE anomalies — deseasoned signal',
              fontsize=8, fontweight='bold', loc='left', pad=3)

leg3 = ax3.legend(loc='upper left', framealpha=0.85,
                  edgecolor='#CCCCCC', fancybox=False,
                  handlelength=1.5, handletextpad=0.5,
                  borderpad=0.5, labelspacing=0.3,
                  ncol=2)
leg3.get_frame().set_linewidth(0.4)

ax3.xaxis.set_major_locator(mdates.YearLocator(5))
ax3.xaxis.set_minor_locator(mdates.YearLocator(1))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# ── Sauvegarde ─────────────────────────────────────────────────────────
OUTPUT_PNG = FIG_DIR + r"\Figure_02_EKE_timeseries.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_02_EKE_timeseries.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()

# ── Caption ───────────────────────────────────────────────────────────────
print("""
CAPTION FIGURE 2:
Figure 2. Temporal variability of basin-averaged eddy kinetic energy (EKE)
in the Algerian Basin derived from CMEMS DUACS DT-2024 all-satellite
altimetry (ALL-SAT), 1993-2023.
(a) Monthly mean EKE time series (thin line and shading), 12-month running
mean (thick line), and linear trend (dashed red line; slope indicated in
legend). Summary statistics are given in the inset box.
(b) Climatological seasonal cycle of EKE averaged over 1993-2023; error
bars indicate the interannual standard deviation for each calendar month.
The month of maximum (August, red bar) and minimum (February, blue bar)
are highlighted.
(c) Deseasoned monthly EKE anomalies; red and blue shading indicate
positive and negative anomalies respectively; the 12-month running mean
(black line) highlights the transition from a low-energy regime prior to
2008 to a sustained high-energy state thereafter (dotted vertical line).
""")
