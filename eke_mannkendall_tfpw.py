"""
EKE TREND ANALYSIS — Mann-Kendall + TFPW
Algerian Basin 1993–2023

Nedjar, Bachari, Houma — Ocean Science (EGU)

Method:
  1. Compute monthly basin-averaged EKE from DUACS TWO-SAT
  2. Deseasonalise (remove monthly climatology)
  3. Pre-whiten using TFPW (Yue et al. 2002)
  4. Apply Mann-Kendall non-parametric trend test
  5. Estimate Sen's slope
  6. Bootstrap power analysis (Monte Carlo, 10,000 replicates)

Key result: τ = 0.099, p = 0.004, Sen's slope = +2.06 cm²/s²/decade
"""

import xarray as xr
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE    = r"C:\Users\kikk\CLAUDE ARTICLE DATA"
F_TWOSAT = (BASE + r"\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
             r"_multi-vars_4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31.nc")

LON_MIN, LON_MAX = -2.0, 9.0
LAT_MIN, LAT_MAX = 35.0, 42.0


def compute_eke(sla_file):
    """Compute monthly basin-averaged EKE from DUACS TWO-SAT."""
    ds = xr.open_dataset(sla_file)
    ugos = ds['ugos'].sel(longitude=slice(LON_MIN, LON_MAX),
                           latitude=slice(LAT_MIN, LAT_MAX))
    vgos = ds['vgos'].sel(longitude=slice(LON_MIN, LON_MAX),
                           latitude=slice(LAT_MIN, LAT_MAX))
    eke_daily = 0.5 * (ugos**2 + vgos**2) * 1e4   # cm²/s²
    eke_monthly = eke_daily.resample(time='MS').mean()
    eke_basin = eke_monthly.mean(dim=['latitude','longitude']).values
    time = eke_monthly.time.values
    ds.close()
    return eke_basin, time


def deseasonalise(eke, time):
    """Remove monthly climatology."""
    months = np.array([int(str(t)[5:7]) for t in time])
    eke_anom = np.zeros_like(eke)
    for m in range(1, 13):
        idx = months == m
        eke_anom[idx] = eke[idx] - eke[idx].mean()
    return eke_anom


def tfpw(x):
    """
    Trend-Free Pre-Whitening (Yue et al. 2002).
    Returns: pre-whitened series, lag-1 autocorrelation r1.
    """
    n = len(x)
    # Step 1: Compute Sen's slope on original series
    slope = stats.theilslopes(x, np.arange(n)).slope
    # Step 2: Remove trend
    x_detrended = x - slope * np.arange(n)
    # Step 3: Estimate lag-1 autocorrelation
    r1 = np.corrcoef(x_detrended[:-1], x_detrended[1:])[0, 1]
    # Step 4: Pre-whiten
    x_pw = x_detrended[1:] - r1 * x_detrended[:-1]
    # Step 5: Restore trend
    x_tfpw = x_pw + slope * np.arange(1, n)
    return x_tfpw, r1


def mann_kendall(x):
    """Non-parametric Mann-Kendall trend test."""
    n = len(x)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    # Variance
    var_s = n * (n - 1) * (2 * n + 5) / 18
    # Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return tau, p, z


def sens_slope_per_decade(x, time_years):
    """Sen's slope in original units per decade."""
    res = stats.theilslopes(x, time_years)
    return res.slope * 10   # per decade


def bootstrap_power(x, tau_obs, n_bootstrap=10000, alpha=0.05):
    """
    Monte Carlo bootstrap power analysis.
    Estimates probability of detecting the observed trend
    given the null hypothesis H0: no trend.
    Post-TFPW residuals used as bootstrap population.
    """
    n = len(x)
    # Pre-whiten to remove trend for null distribution
    x_pw, _ = tfpw(x)
    # Bootstrap
    n_sig = 0
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(x_pw, size=n, replace=True)
        tau_b, p_b, _ = mann_kendall(sample)
        if p_b < alpha:
            n_sig += 1
    power = n_sig / n_bootstrap
    return power


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("EKE Trend Analysis — Algerian Basin 1993–2023")
    print("=" * 55)

    # 1. Compute EKE
    eke, time = compute_eke(F_TWOSAT)
    years = np.array([1993 + i/12 for i in range(len(time))])
    print(f"  EKE series : {len(eke)} months  "
          f"mean = {np.mean(eke):.1f} ± {np.std(eke):.1f} cm²/s²")

    # 2. Deseasonalise
    eke_anom = deseasonalise(eke, time)

    # 3. TFPW
    eke_pw, r1 = tfpw(eke_anom)
    print(f"  Lag-1 autocorrelation r1 = {r1:.3f}")

    # 4. Mann-Kendall on TFPW series
    tau, p, z = mann_kendall(eke_pw)
    print(f"\n  Mann-Kendall (post-TFPW) :")
    print(f"    τ = {tau:.3f}")
    print(f"    p = {p:.4f}  {'*** SIGNIFICANT' if p < 0.05 else 'NS'}")
    print(f"    z = {z:.3f}")

    # 5. Sen's slope
    slope_dec = sens_slope_per_decade(eke_anom, years)
    print(f"\n  Sen's slope = {slope_dec:+.2f} cm²/s²/decade")

    # 6. Power analysis
    print(f"\n  Bootstrap power analysis (n=10,000)...")
    power = bootstrap_power(eke_pw, tau)
    print(f"    Statistical power ≈ {power:.2f}")

    print(f"\n  Key result for manuscript:")
    print(f"    +{slope_dec:.2f} cm²/s²/decade  (τ={tau:.3f}, p={p:.3f})")
    print(f"    r1={r1:.3f}  power≈{power:.2f}")
