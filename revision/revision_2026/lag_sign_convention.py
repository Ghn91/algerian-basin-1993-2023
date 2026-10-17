"""Sign convention of the lag in the two cross-correlation routines of the
original analysis (src/fig08_wemo_era5_crosscorr.py), established on series
whose lead is known by construction.

Usage: python lag_sign_convention.py
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

AVANCE = 3
N = 372
BRUIT = 0.05

ETIQUETTE_PUBLIEE = "negative = WeMO leads EKE"


def _lag_fig08(w, e, max_lag=24):
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = np.zeros(len(lags))
    for k, lag in enumerate(lags):
        if lag == 0:
            r, _ = stats.pearsonr(w, e)
        elif lag > 0:
            r, _ = stats.pearsonr(w[:-lag], e[lag:])
        else:
            r, _ = stats.pearsonr(w[-lag:], e[:lag])
        corrs[k] = r
    i = int(np.argmax(np.abs(corrs)))
    return int(lags[i]), float(corrs[i])


def _lag_ub(eke_d, eke_v, idx_d, idx_v, max_lag=24):
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        idx_d_shifted = idx_d.shift(lag)
        common = np.intersect1d(eke_d, idx_d_shifted)
        if len(common) < 30:
            continue
        x = eke_v[np.isin(eke_d, common)]
        y = idx_v[np.isin(idx_d_shifted, common)]
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 30:
            continue
        r, _ = stats.pearsonr(x[m], y[m])
        rows.append({"lag": lag, "r": r})
    df = pd.DataFrame(rows)
    pk = df.loc[df["r"].abs().idxmax()]
    return int(pk["lag"]), float(pk["r"])


def fabriquer(avance, graine=0, n=N, bruit=BRUIT):
    rng = np.random.default_rng(graine)
    w = rng.normal(size=n)
    e = np.roll(w, avance)
    if avance > 0:
        e[:avance] = rng.normal(size=avance)
    elif avance < 0:
        e[avance:] = rng.normal(size=-avance)
    e = e + bruit * rng.normal(size=n)
    return w, e, pd.period_range("1993-01", periods=n, freq="M")


def mesurer(avance, graine=0):
    w, e, d = fabriquer(avance, graine)
    l8, r8 = _lag_fig08(w, e)
    lu, ru = _lag_ub(d, e, d, w)
    return (l8, r8), (lu, ru)


def etiquette_est_juste(etiquette=ETIQUETTE_PUBLIEE):
    (l8, _), _ = mesurer(+AVANCE)
    code_dit_negatif = l8 < 0
    etiquette_dit_negatif = (
        "negative" in etiquette.lower() and "leads" in etiquette.lower()
    )
    return code_dit_negatif == etiquette_dit_negatif, l8


def rapport():
    (l8, r8), (lu, ru) = mesurer(+AVANCE)

    juste, _ = etiquette_est_juste()


if __name__ == "__main__":
    sys.exit(rapport() or 0)
