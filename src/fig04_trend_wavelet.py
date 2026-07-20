# Figure 4 - Mann-Kendall / Sen trend and Morlet wavelet (TWO-SAT)
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, BASE

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
import pymannkendall as mk
import pywt
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

# ── Chargement TWO-SAT ────────────────────────────────────────────────────
print("Chargement TWO-SAT...")
ds2 = xr.open_dataset(
    DATA_DIR + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg"
               r"_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc"
)

# ── EKE TWO-SAT ───────────────────────────────────────────────────────────
print("Calcul EKE TWO-SAT...")
ugos_mean_t = ds2['ugos'].mean('time')
vgos_mean_t = ds2['vgos'].mean('time')
ugosa2 = ds2['ugos'] - ugos_mean_t
vgosa2 = ds2['vgos'] - vgos_mean_t
eke2_daily   = (0.5*(ugosa2**2 + vgosa2**2)*1e4
                ).mean(dim=['latitude','longitude'], skipna=True)
eke2_monthly = eke2_daily.resample(time='ME').mean()
eke2_clim    = eke2_monthly.groupby('time.month').mean()
eke2_anom    = eke2_monthly.groupby('time.month') - eke2_clim

vals    = eke2_monthly.values
vals_an = eke2_anom.values
times_m = eke2_monthly.time.values
N       = len(vals)
ds2.close()

# ── Mann-Kendall ──────────────────────────────────────────────────────────
print("Test de Mann-Kendall...")
result_mk   = mk.original_test(vals)
slope_decade= result_mk.slope * 12 * 10
t_num       = np.arange(N)
trend_mk    = result_mk.slope * t_num + np.nanmedian(vals)
smooth12    = uniform_filter1d(vals, size=12, mode='nearest')

# ── Wavelet Morlet ────────────────────────────────────────────────────────
print("Analyse Wavelet...")
signal  = vals_an - np.nanmean(vals_an)
signal  = np.where(np.isnan(signal), 0, signal)
scales  = np.arange(2, 130)
freqs   = pywt.scale2frequency('cmor1.5-1.0', scales) / 1.0
periods = 1.0 / freqs
coeffs, _ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1.0)
power      = np.abs(coeffs)**2
power_norm = power / power.max()
coi        = np.minimum(np.arange(1, N+1), np.arange(N, 0, -1)) * 2.0
global_pwr = np.nanmean(power, axis=1)
peak_idx   = np.argmax(global_pwr)
print(f"Pic dominant : {periods[peak_idx]:.0f} mois")
print(f"Mann-Kendall : p={result_mk.p:.4f} | slope={slope_decade:+.2f} cm2/s2/dec")

# ── Figure 3 ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.0, 7.2))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(
    3, 1,
    figure=fig,
    hspace=0.42,
    left=0.10, right=0.97,
    top=0.96,  bottom=0.07
)

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
# PANNEAU (a) — EKE TWO-SAT + Mann-Kendall
# ─────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1)

ax1.fill_between(times_m, vals,
                 alpha=0.25, color='#5BA4CF', linewidth=0)
ax1.plot(times_m, vals,
         color='#5BA4CF', linewidth=0.5, alpha=0.80,
         label='Monthly EKE (TWO-SAT)')
ax1.plot(times_m, smooth12,
         color='#1A5F7A', linewidth=1.8,
         label='12-month running mean')
ax1.plot(times_m, trend_mk,
         color='#C0392B', linewidth=1.5, linestyle='--',
         label=f"Sen's slope: {slope_decade:+.2f} cm$^2$ s$^{{-2}}$ decade$^{{-1}}$")

# Boîte Mann-Kendall — positionnée à droite pour éviter la légende
sig_str = 'significant' if result_mk.p < 0.05 else 'not significant'
mk_txt  = (f'Mann-Kendall test\n'
           f'Trend: {result_mk.trend}  ({sig_str})\n'
           f'$\\tau$ = {result_mk.Tau:.3f}   '
           f'p = {result_mk.p:.4f}')
ax1.text(0.985, 0.97, mk_txt,
         transform=ax1.transAxes,
         fontsize=6.0, va='top', ha='right',
         bbox=dict(facecolor='#FFF9EC', alpha=0.90,
                   edgecolor='#DDAA44', linewidth=0.5,
                   boxstyle='round,pad=0.4'))

ax1.set_xlim(times_m[0], times_m[-1])
ax1.set_ylabel('EKE (cm$^2$ s$^{-2}$)', fontsize=7.5)
ax1.set_title('(a)  EKE trend analysis — TWO-SAT altimetry, Mann-Kendall test (1993-2023)',
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
# PANNEAU (b) — Scalogramme Wavelet
# ─────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)

t_idx = np.arange(N)

cf2 = ax2.contourf(
    t_idx, periods, power_norm,
    levels=np.linspace(0, 1, 25),
    cmap='YlOrRd', extend='max'
)
ax2.contour(
    t_idx, periods, power_norm,
    levels=[0.25, 0.50, 0.75],
    colors='black', linewidths=0.35, alpha=0.55
)

# Cône d'influence
ax2.fill_between(t_idx, coi, periods.max(),
                 alpha=0.25, color='#888888',
                 hatch='///', linewidth=0,
                 label="Cone of influence (COI)")
ax2.plot(t_idx, coi, color='#555555',
         linewidth=0.8, linestyle='--')

# Lignes de référence
ref_periods = [(12,'1 yr'), (24,'2 yr'), (60,'5 yr'), (120,'10 yr')]
for p_ref, lbl in ref_periods:
    if p_ref <= periods.max():
        ax2.axhline(p_ref, color='#2255AA', linewidth=0.55,
                    linestyle=':', alpha=0.75)
        ax2.text(N*0.985, p_ref*1.04, lbl,
                 ha='right', va='bottom',
                 fontsize=5.8, color='#2255AA')

# Axes
year_ticks  = np.arange(0, N, 60)
year_labels = [str(1993 + i//12) for i in year_ticks]
ax2.set_xticks(year_ticks)
ax2.set_xticklabels(year_labels, fontsize=6.5)
ax2.set_yscale('log')
ax2.set_ylim(2, min(130, periods.max()))
ax2.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
ax2.set_ylabel('Period (months)', fontsize=7.5)
ax2.set_xlabel('Year', fontsize=7.5)
ax2.set_title('(b)  Morlet wavelet scalogram — deseasoned EKE anomaly',
              fontsize=8, fontweight='bold', loc='left', pad=3)
ax2.legend(loc='upper right', framealpha=0.80,
           edgecolor='#CCCCCC', fancybox=False,
           fontsize=6.0).get_frame().set_linewidth(0.4)

# Colorbar scalogramme
pos2 = ax2.get_position()
cax2 = fig.add_axes([pos2.x1 + 0.010, pos2.y0 + 0.01,
                     0.013, pos2.height - 0.02])
cb2  = fig.colorbar(cf2, cax=cax2)
cb2.set_label('Normalised power', fontsize=6.0, labelpad=2)
cb2.ax.tick_params(labelsize=5.5, width=0.4, length=2)
cb2.outline.set_linewidth(0.4)
cb2.set_ticks([0, 0.25, 0.50, 0.75, 1.0])

# ─────────────────────────────────────────────────────────────────────────
# PANNEAU (c) — Spectre de puissance global
# ─────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, xlabel=True)

ax3.plot(periods, global_pwr,
         color='#1A5F7A', linewidth=1.8, zorder=4)
ax3.fill_between(periods, global_pwr,
                 alpha=0.25, color='#5BA4CF', linewidth=0)

# Lignes de référence
ref_cols = ['#2ECC71','#E67E22','#E74C3C','#9B59B6']
ref_lbls = ['Annual (12 m)','Bi-annual (24 m)',
            'Pentadal (60 m)','Decadal (120 m)']
for p_ref, col, lbl in zip([12,24,60,120], ref_cols, ref_lbls):
    if p_ref <= periods.max():
        ax3.axvline(p_ref, color=col, linewidth=1.2,
                    linestyle='--', alpha=0.80, label=lbl, zorder=3)

# Annoter le pic dominant
peak_p = periods[peak_idx]
peak_v = global_pwr[peak_idx]
ax3.annotate(
    f'Dominant peak\n{peak_p:.0f} months (~{peak_p/12:.1f} yr)',
    xy=(peak_p, peak_v),
    xytext=(peak_p + 15, peak_v * 0.82),
    fontsize=6.5, color='#C0392B', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.9)
)
# Marquer le pic
ax3.plot(peak_p, peak_v, 'o',
         color='#C0392B', markersize=5, zorder=5)

ax3.set_xlim(2, min(130, periods.max()))
ax3.set_xlabel('Period (months)', fontsize=7.5)
ax3.set_ylabel('Power (cm$^4$ s$^{-4}$)', fontsize=7.5)
ax3.set_title('(c)  Global wavelet power spectrum',
              fontsize=8, fontweight='bold', loc='left', pad=3)
leg3 = ax3.legend(loc='upper left', framealpha=0.85,
                  edgecolor='#CCCCCC', fancybox=False,
                  handlelength=1.5, handletextpad=0.5,
                  borderpad=0.5, labelspacing=0.3,
                  ncol=2)
leg3.get_frame().set_linewidth(0.4)

# ── Sauvegarde ────────────────────────────────────────────────────────────
OUTPUT_PNG = FIG_DIR + r"\Figure_03_MannKendall_Wavelet.png"
OUTPUT_PDF = FIG_DIR + r"\Figure_03_MannKendall_Wavelet.pdf"

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(OUTPUT_PDF, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"PNG -> {OUTPUT_PNG}")
print(f"PDF -> {OUTPUT_PDF}")
plt.show()

print("""
CAPTION FIGURE 3:
Figure 3. Trend and multi-scale variability analysis of basin-averaged EKE
in the Algerian Basin, derived from the climate-grade two-satellite altimetry
product (TWO-SAT, CMEMS DUACS DT-2024), 1993-2023.
(a) Monthly mean EKE time series (thin line and shading), 12-month running
mean (thick blue line), and Sen's slope estimate (dashed red line). The
Mann-Kendall test result is summarized in the inset box.
(b) Continuous Morlet wavelet scalogram of the deseasoned monthly EKE anomaly
time series. Colour intensity represents normalised wavelet power. Black
contours mark the 25th, 50th, and 75th power percentiles. The hatched
region indicates the cone of influence (COI) where edge effects reduce
reliability. Horizontal dotted lines indicate reference periods of 1, 2,
5, and 10 years.
(c) Global wavelet power spectrum (time-averaged wavelet power). Vertical
dashed lines indicate reference periodicities. The dominant spectral peak
at 73 months (~6.1 years) is annotated.
""")

# 95% confidence interval on Sen's slope (Theil-Sen)
if __name__ == "__main__":
    import xarray as _xr, numpy as _np
    from scipy.stats import theilslopes as _ts
    _f = (DATA_DIR + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg"
          r"_P1D_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc")
    _ds = _xr.open_dataset(_f)
    _um, _vm = _ds['ugos'].mean('time'), _ds['vgos'].mean('time')
    _eke = (0.5*((_ds['ugos']-_um)**2 + (_ds['vgos']-_vm)**2)*1e4
            ).mean(dim=['latitude','longitude'], skipna=True).resample(time='ME').mean()
    _ds.close()
    _y = _np.asarray(_eke.values, float)
    _sl, _ic, _lo, _hi = _ts(_y, _np.arange(len(_y)), 0.95)
    print(f"Sen slope = {_sl*120:+.3f} cm2/s2/decade "
          f"(95% CI: [{_lo*120:+.2f}, {_hi*120:+.2f}])")

