"""Morlet wavelet spectrum of the EKE anomaly with red-noise significance, the
lag-one autocorrelation measured on the series, and the false-alarm rate of
the test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

RACINE = Path(__file__).resolve().parents[2]
sys.path.append(str(RACINE))
sys.path.append(str(RACINE / "src"))

import pandas as pd
from provenance import sauver_table

DERIVE_EKE = RACINE / "data" / "interim" / "eke_mensuelle_twosat.csv"
TAB04 = (
    RACINE / "tables" / "ch03" / "ch03_tab04_significativite_des_pics_d_ondelettes.csv"
)
ARCHIVE = RACINE / "docs" / "ch03_ondelettes.json"

OMEGA0 = 6.0
DJ = 0.125
PERIODES_ARTICLE = (2, 130)
PICS_ANNONCES = (74, 24)

ALPHA_MANUSCRIT = 0.115
BALAYAGE_ALPHA = (0.0, 0.115, 0.2, 0.3, 0.4, 0.5, 0.6, 0.677, 0.7, 0.8)


def _ondelettes(x, dt=1.0):
    n = len(x)
    x = x - x.mean()
    s0 = 2 * dt
    J = int(np.log2(n * dt / s0) / DJ)
    s = s0 * 2.0 ** (np.arange(J + 1) * DJ)

    xf = np.fft.fft(x)
    k = np.fft.fftfreq(n, d=dt) * 2 * np.pi

    W = np.zeros((len(s), n), dtype=complex)
    for i, si in enumerate(s):
        expo = -((si * k - OMEGA0) ** 2) / 2.0
        psi = (np.pi**-0.25) * np.sqrt(2 * np.pi * si / dt) * np.exp(expo)
        psi[k <= 0] = 0.0
        W[i] = np.fft.ifft(xf * psi)

    lam = 4 * np.pi * s / (OMEGA0 + np.sqrt(2 + OMEGA0**2))
    return np.abs(W) ** 2, lam, s


def _fond_rouge(alpha, lam, dt=1.0):
    f = dt / lam
    return (1 - alpha**2) / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * f))


def _seuils(x, lam, s, dt=1.0, conf=0.95, alpha=None):
    n = len(x)
    a1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if alpha is None else float(alpha)
    var = x.var(ddof=1)
    fond = _fond_rouge(a1, lam, dt) * var

    seuil_local = fond * chi2.ppf(conf, 2) / 2.0
    ddl = 2 * np.sqrt(1 + (n * dt / (2.32 * s)) ** 2)
    seuil_global = fond * chi2.ppf(conf, ddl) / ddl
    return a1, fond, seuil_local, seuil_global


def _coi(n, s, lam, dt=1.0):
    coef = lam[0] / s[0] * np.sqrt(2)
    t = np.arange(n) * dt
    return coef * np.minimum(t, (n - 1) * dt - t)


def analyser(x, dt=1.0, alpha=None, bavard=False):
    x = np.asarray(x, dtype=float)
    puis, lam, s = _ondelettes(x, dt)
    a1, fond, s_loc, s_glo = _seuils(x, lam, s, dt, alpha=alpha)
    coi = _coi(len(x), s, lam, dt)

    masque = lam[:, None] <= coi[None, :]
    compte = masque.sum(axis=1)
    somme = np.where(masque, puis, 0.0).sum(axis=1)
    global_spec = np.divide(
        somme, compte, out=np.full_like(somme, np.nan), where=compte > 0
    )

    dans = (lam >= PERIODES_ARTICLE[0]) & (lam <= PERIODES_ARTICLE[1])
    return dict(
        lam=lam,
        global_spec=global_spec,
        seuil=s_glo,
        dans=dans,
        alpha=a1,
        coi=coi,
        hors_cone=int(masque.sum()),
    )


def rapport(res, pics=PICS_ANNONCES):
    lam, g, seuil, dans = (res["lam"], res["global_spec"], res["seuil"], res["dans"])
    lignes = []
    for p in pics:
        i = int(np.nanargmin(np.abs(lam - p)))
        passe = bool(np.isfinite(g[i]) and g[i] >= seuil[i])
        lignes.append(
            dict(
                periode_annoncee_mois=str(p),
                periode_analysee_mois=float(lam[i]),
                puissance=float(g[i]),
                seuil_95=float(seuil[i]),
                rapport_au_seuil=float(g[i] / seuil[i]),
                franchit="oui" if passe else "NON",
            )
        )
    gd = np.where(dans, g, np.nan)
    imax = int(np.nanargmax(gd))
    lignes.append(
        dict(
            periode_annoncee_mois="maximum observe",
            periode_analysee_mois=float(lam[imax]),
            puissance=float(g[imax]),
            seuil_95=float(seuil[imax]),
            rapport_au_seuil=float(g[imax] / seuil[imax]),
            franchit="oui" if g[imax] >= seuil[imax] else "NON",
        )
    )
    return lignes


PERIODE_DU_PIC = 50.0
TIRAGES_PLANCHER = 20


def bruit_rouge(phi, n, graine):
    rng = np.random.default_rng(graine)
    y = np.zeros(n)
    innovation = np.sqrt(1.0 - phi**2)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + innovation * rng.normal()
    return y


def plancher(
    phi, alpha, n=372, tirages=TIRAGES_PLANCHER, periode_min=PERIODE_DU_PIC, depart=5000
):
    taux = []
    for g in range(tirages):
        r = analyser(bruit_rouge(phi, n, depart + g), alpha=alpha)
        d = r["dans"] & np.isfinite(r["global_spec"])
        if periode_min is not None:
            d = d & (r["lam"] >= periode_min)
        taux.append(
            float(np.sum((r["global_spec"] >= r["seuil"]) & d) / max(int(d.sum()), 1))
        )
    return float(np.mean(taux))


def alpha_de_bascule(x, pic, balayage=BALAYAGE_ALPHA):
    passants = []
    for a in balayage:
        r = analyser(x, alpha=a)
        i = int(np.nanargmin(np.abs(r["lam"] - pic)))
        if np.isfinite(r["global_spec"][i]) and r["global_spec"][i] >= r["seuil"][i]:
            passants.append(a)
    return max(passants) if passants else None


def lire_eke(chemin=DERIVE_EKE):
    p = Path(chemin)
    if not p.exists():
        raise SystemExit(
            f"the monthly EKE series is missing: {p}\nRun first:\n  python src/ch03_altimetrie/ch03_tab20_correlation_wemo.py"
        )
    d = pd.read_csv(p)
    return list(d["mois"]), np.asarray(d["eke_cm2_s2"], dtype=float)


def desaisonnaliser(mois, valeurs):
    v = np.asarray(valeurs, dtype=float)
    cal = np.array([int(m[5:7]) for m in mois])
    sortie = v.copy()
    for m in range(1, 13):
        sel = cal == m
        if sel.any():
            sortie[sel] = v[sel] - v[sel].mean()
    return sortie


def produire():
    mois, brute = lire_eke()
    anomalie = desaisonnaliser(mois, brute)
    a1_brute = float(np.corrcoef(brute[:-1], brute[1:])[0, 1])
    a1_anom = float(np.corrcoef(anomalie[:-1], anomalie[1:])[0, 1])

    lignes22, lignes23 = [], []
    for etiquette, alpha in (
        ("alpha mesure sur l'anomalie", None),
        ("alpha declare au manuscrit", ALPHA_MANUSCRIT),
    ):
        res = analyser(anomalie, alpha=alpha, bavard=True)
        for L in rapport(res):
            lignes22.append(
                {"fond_rouge": etiquette, "alpha_employe": res["alpha"], **L}
            )

    for a in BALAYAGE_ALPHA:
        res = analyser(anomalie, alpha=a)
        ligne = {"alpha": a}
        for p in PICS_ANNONCES:
            i = int(np.nanargmin(np.abs(res["lam"] - p)))
            r = float(res["global_spec"][i] / res["seuil"][i])
            ligne[f"rapport_au_seuil_{p}_mois"] = r
            ligne[f"franchit_{p}_mois"] = "oui" if r >= 1.0 else "non"
        ligne["plancher_grandes_periodes_pourcent"] = 100 * plancher(a1_anom, a)
        ligne["plancher_tout_le_spectre_pourcent"] = 100 * plancher(
            a1_anom, a, periode_min=None
        )
        lignes23.append(ligne)
    bascule = alpha_de_bascule(anomalie, PICS_ANNONCES[0])

    ecart = None
    if TAB04.exists():
        aout = pd.read_csv(TAB04)
        colonnes = [c for c in aout.columns if "rapport" in c.lower()]
        if colonnes:
            ref = [float(v) for v in aout[colonnes[0]] if str(v).strip()]
            ici = [
                L["rapport_au_seuil"]
                for L in lignes22
                if L["fond_rouge"] == "alpha mesure sur l'anomalie"
            ]
            if ref and ici:
                ecart = float(abs(ref[0] - ici[0]))

    SCRIPT = "src/ch03_altimetrie/ch03_tab22_ondelettes.py"
    DOM = "bassin-algerien-1993-2023"
    SRC = "eke_mensuelle_twosat.csv"

    def texte(rows):
        return pd.DataFrame(
            [
                {c: (f"{v:.6f}" if isinstance(v, float) else v) for c, v in L.items()}
                for L in rows
            ]
        )

    sauver_table(
        texte(lignes22),
        "ch03_tab22",
        "significativite des pics d ondelettes par la chaine de these",
        SCRIPT,
        SRC,
        "3.7",
        domaine=DOM,
    )
    sauver_table(
        texte(lignes23),
        "ch03_tab23",
        "sensibilite du seuil d ondelettes a l alpha du fond rouge",
        SCRIPT,
        SRC,
        "3.7",
        domaine=DOM,
    )

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_text(
        json.dumps(
            {
                "implementation": "constantes du test d'ondelettes de la version soumise",
                "reference": "Torrence & Compo (1998), eq. 3, 6, 12, 18, 23",
                "constantes": {
                    "omega0": OMEGA0,
                    "dj": DJ,
                    "periodes_mois": list(PERIODES_ARTICLE),
                    "pics_annonces_mois": list(PICS_ANNONCES),
                },
                "alphas": {
                    "serie_brute": a1_brute,
                    "anomalie": a1_anom,
                    "declare_au_manuscrit": ALPHA_MANUSCRIT,
                },
                "alpha_de_bascule_du_pic_principal": bascule,
                "ecart_au_recalcul_d_aout": ecart,
                "mois": len(mois),
                "premier": mois[0],
                "dernier": mois[-1],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return lignes22, lignes23


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    a = ap.parse_args()
    produire()
