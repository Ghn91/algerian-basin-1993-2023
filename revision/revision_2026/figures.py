"""Figures 1 to 9 of the revised manuscript, generated from the versioned
derived files, outputs and tables of this archive. Each figure checks that the
quantities stated in the text are those it plots before writing.

Usage: python figures.py --figure N [--apercu DIR] | --toutes
"""

import csv
import io
import os
import sys

import numpy as np

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
CIBLE = os.path.join(DOSSIER, "manuscript")
ATLAS = {
    "anticyclone": os.path.join(
        RACINE, "origine_2026-03", "anticyclones_1993_2023_ALL.csv"
    ),
    "cyclone": os.path.join(RACINE, "origine_2026-03", "cyclones_1993_2023_ALL.csv"),
}
EMPRISE = (-5.0, 10.0, 35.0, 41.0)
LARGEUR_MM = 190.0
LARGEUR_POUCES = LARGEUR_MM / 25.4
DPI = 300
ROUGE, BLEU = "#b2182b", "#2166ac"
COULEUR = {"anticyclone": ROUGE, "cyclone": BLEU}
NOM = {"anticyclone": "Anticyclones", "cyclone": "Cyclones"}
GRIS_SEUIL = "0.35"
ISOBATHES = (("K_200", 200), ("J_1000", 1000), ("I_2000", 2000))
LARGEUR_HEXAGONE_DEG = 0.25
MOIS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
MOIS_EN = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
POLICES = ["Arial", "Liberation Sans", "DejaVu Sans"]


def lire_csv(chemin):
    with io.open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lire_atlas(chemin):
    d = {
        k: []
        for k in (
            "date",
            "annee",
            "mois",
            "lon",
            "lat",
            "amplitude_cm",
            "rayon_km",
            "vitesse_cm_s",
        )
    }
    with io.open(chemin, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d["date"].append(r["date"])
            d["annee"].append(int(r["date"][:4]))
            d["mois"].append(int(r["date"][5:7]))
            d["lon"].append(float(r["lon"]))
            d["lat"].append(float(r["lat"]))
            d["amplitude_cm"].append(float(r["amplitude"]) * 100.0)
            d["rayon_km"].append(float(r["radius_s"]) / 1000.0)
            d["vitesse_cm_s"].append(float(r["speed_average"]) * 100.0)
    return {k: (np.array(v) if k != "date" else v) for k, v in d.items()}


def comptes_annuels(annees, debut=1993, fin=2023):
    ans = np.arange(debut, fin + 1)
    return ans, np.array([(annees == a).sum() for a in ans])


def cycle_saisonnier(mois, n_annees):
    return np.array([(mois == m).sum() for m in range(1, 13)]) / float(n_annees)


def extremes(ans, n):
    i, j = int(np.argmin(n)), int(np.argmax(n))
    egalites = int((n == n[i]).sum() > 1 or (n == n[j]).sum() > 1)
    return int(ans[i]), int(n[i]), int(ans[j]), int(n[j]), egalites


def pas_des_fleches(coord, espacement_deg=0.5):
    pas = abs(float(np.median(np.diff(coord))))
    return max(1, int(round(espacement_deg / pas)))


def borne_symetrique(valeurs, marge=1.05):
    return float(np.nanmax(np.abs(valeurs))) * marge


def bloc_retenu(lignes, regle):
    couv = {int(l["bloc_mois"]): float(l["couverture_corrigee"]) for l in lignes}
    fact = {int(l["bloc_mois"]): float(l["facteur_inflation"]) for l in lignes}
    return regle(couv, fact)


def classes_visibles(classes, aires):
    return [c for c, a in zip(classes, aires) if a > 0.0]


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    dispo = {f.name for f in font_manager.fontManager.ttflist}
    police = next((p for p in POLICES if p in dispo), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [police],
            "mathtext.fontset": "custom",
            "mathtext.rm": police,
            "mathtext.it": f"{police}:italic",
            "mathtext.bf": f"{police}:bold",
            "mathtext.default": "regular",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.5,
            "hatch.linewidth": 0.4,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.4,
            "legend.columnspacing": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "figure.constrained_layout.h_pad": 0.04,
            "figure.constrained_layout.w_pad": 0.04,
            "figure.constrained_layout.hspace": 0.03,
            "figure.constrained_layout.wspace": 0.03,
        }
    )
    return plt


def nouvelle_figure(plt, hauteur_mm):
    return plt.figure(figsize=(LARGEUR_POUCES, hauteur_mm / 25.4), layout="constrained")


def lettre(ax, l):
    from matplotlib.transforms import ScaledTranslation

    t = ax.transAxes + ScaledTranslation(0.0, 2.5 / 72.0, ax.figure.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        f"({l})",
        transform=t,
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
    )


def fond_de_carte(
    ax, ccrs, emprise=EMPRISE, gauche=True, bas=True, isobathes=True, zorder_terre=3
):
    import cartopy.io.shapereader as sr
    from shapely.geometry import box

    ax.set_extent(emprise, crs=ccrs.PlateCarree())
    cadre = box(emprise[0] - 0.5, emprise[2] - 0.5, emprise[1] + 0.5, emprise[3] + 0.5)
    if isobathes:
        for nom, _ in ISOBATHES:
            geo = [
                g.intersection(cadre)
                for g in sr.Reader(
                    sr.natural_earth(
                        resolution="10m", category="physical", name=f"bathymetry_{nom}"
                    )
                ).geometries()
                if g.intersects(cadre)
            ]
            ax.add_geometries(
                geo,
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="0.5",
                linewidth=0.3,
                zorder=2,
            )
    terre = [
        g.intersection(cadre)
        for g in sr.Reader(
            sr.natural_earth(resolution="10m", category="physical", name="land")
        ).geometries()
        if g.intersects(cadre)
    ]
    ax.add_geometries(
        terre,
        ccrs.PlateCarree(),
        facecolor="0.88",
        edgecolor="0.25",
        linewidth=0.35,
        zorder=zorder_terre,
    )
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        color="0.6",
        linestyle=":",
        xlocs=[-4, 0, 4, 8],
        ylocs=[36, 38, 40],
    )
    gl.top_labels = gl.right_labels = False
    gl.left_labels, gl.bottom_labels = gauche, bas
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    gl.xpadding = gl.ypadding = 2
    ax.spines["geo"].set_linewidth(0.5)
    return gl


def barre(fig, mappable, ax, label, lieu="bottom", **kw):
    defaut = dict(
        location=lieu, shrink=1.0, aspect=35 if lieu == "bottom" else 14, pad=0.02
    )
    defaut.update(kw)
    cb = fig.colorbar(mappable, ax=ax, **defaut)
    cb.set_label(label)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    return cb


class Debordement(Exception):
    pass


def ecrire(fig, nom, apercu=None):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    w, h = fig.get_size_inches()
    tol = 0.02
    if bb.x0 < -tol or bb.y0 < -tol or bb.x1 > w + tol or bb.y1 > h + tol:
        raise Debordement(
            f"{nom}: content outside the figure ({bb.x0:.2f}, {bb.y0:.2f}, {bb.x1:.2f}, {bb.y1:.2f}) for {w:.2f} x {h:.2f} inches"
        )
    dossier = apercu or CIBLE
    os.makedirs(dossier, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dossier, f"{nom}.{ext}"), format=ext, dpi=DPI)
    from PIL import Image

    largeur = Image.open(os.path.join(dossier, f"{nom}.png")).size[0]
    attendu = int(round(LARGEUR_POUCES * DPI))
    if abs(largeur - attendu) > 1:
        raise Debordement(f"{nom}: {largeur} px wide, {attendu} expected")


CLASSES_BATHY = (
    ("L_0", 0),
    ("K_200", 200),
    ("J_1000", 1000),
    ("I_2000", 2000),
    ("H_3000", 3000),
)
EMPRISE_SITUATION = (-6.5, 11.0, 34.3, 42.3)
EMPRISE_MEDITERRANEE = (-7.0, 37.0, 29.5, 46.5)
LIEUX = (
    ("ALGERIA", 2.5, 34.7, 7, "bold"),
    ("MOROCCO", -4.6, 34.7, 6.5, "bold"),
    ("TUNISIA", 9.35, 35.55, 6.5, "bold"),
    ("SPAIN", -3.0, 38.6, 6.5, "bold"),
    ("Algerian Basin", 4.5, 38.3, 7, "normal"),
    ("Alboran Sea", -3.4, 36.05, 6, "normal"),
    ("Balearic Islands", 1.6, 40.15, 6, "normal"),
    ("Sardinia", 9.05, 40.05, 5.5, "normal"),
    ("Sardinia Channel", 9.55, 37.65, 5.5, "normal"),
)


def figure_1(apercu=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as sr
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle
    from shapely.geometry import box

    plt = style()
    proj = ccrs.PlateCarree()
    cadre = box(
        EMPRISE_SITUATION[0] - 0.5,
        EMPRISE_SITUATION[2] - 0.5,
        EMPRISE_SITUATION[1] + 0.5,
        EMPRISE_SITUATION[3] + 0.5,
    )
    vue = box(
        EMPRISE_SITUATION[0],
        EMPRISE_SITUATION[2],
        EMPRISE_SITUATION[1],
        EMPRISE_SITUATION[3],
    )
    geos, aires = [], []
    for nom, _ in CLASSES_BATHY:
        g = [
            x.intersection(cadre)
            for x in sr.Reader(
                sr.natural_earth(
                    resolution="10m", category="physical", name=f"bathymetry_{nom}"
                )
            ).geometries()
            if x.intersects(cadre)
        ]
        geos.append(g)
        aires.append(sum(x.intersection(vue).area for x in g))
    visibles = classes_visibles(list(range(len(CLASSES_BATHY))), aires)
    bleus = plt.get_cmap("Blues")(np.linspace(0.10, 0.80, len(visibles)))

    l_carte = 0.745 * LARGEUR_MM
    h_carte = (
        l_carte
        * (EMPRISE_SITUATION[3] - EMPRISE_SITUATION[2])
        / (EMPRISE_SITUATION[1] - EMPRISE_SITUATION[0])
    )
    h_fig = h_carte + 9.0
    fig = plt.figure(figsize=(LARGEUR_POUCES, h_fig / 25.4))
    ax = fig.add_axes([0.055, 7.0 / h_fig, 0.745, h_carte / h_fig], projection=proj)
    ax.set_extent(EMPRISE_SITUATION, crs=proj)
    for i, c in zip(visibles, bleus):
        ax.add_geometries(geos[i], proj, facecolor=c, edgecolor="none", zorder=1)
    terre = [
        g.intersection(cadre)
        for g in sr.Reader(
            sr.natural_earth(resolution="10m", category="physical", name="land")
        ).geometries()
        if g.intersects(cadre)
    ]
    ax.add_geometries(
        terre, proj, facecolor="0.88", edgecolor="0.25", linewidth=0.35, zorder=3
    )
    ax.add_patch(
        Rectangle(
            (EMPRISE[0], EMPRISE[2]),
            EMPRISE[1] - EMPRISE[0],
            EMPRISE[3] - EMPRISE[2],
            fill=False,
            edgecolor=ROUGE,
            linewidth=1.0,
            transform=proj,
            zorder=5,
        )
    )
    for nom, x, y, taille, graisse in LIEUX:
        ax.text(
            x,
            y,
            nom,
            fontsize=taille,
            fontweight=graisse,
            ha="center",
            va="center",
            transform=proj,
            zorder=6,
            color="0.15",
            style="italic" if graisse == "normal" else "normal",
        )
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        color="0.6",
        linestyle=":",
        xlocs=range(-6, 12, 2),
        ylocs=range(35, 43, 1),
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 6.5}
    ax.spines["geo"].set_linewidth(0.5)

    l_ins = 0.165 * LARGEUR_MM
    h_ins = (
        l_ins
        * (EMPRISE_MEDITERRANEE[3] - EMPRISE_MEDITERRANEE[2])
        / (EMPRISE_MEDITERRANEE[1] - EMPRISE_MEDITERRANEE[0])
    )
    ins = fig.add_axes(
        [0.825, (7.0 + h_carte - h_ins) / h_fig, 0.165, h_ins / h_fig], projection=proj
    )
    ins.set_extent(EMPRISE_MEDITERRANEE, crs=proj)
    ins.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "50m"),
        facecolor="0.85",
        edgecolor="0.3",
        linewidth=0.25,
        zorder=1,
    )
    ins.add_patch(
        Rectangle(
            (EMPRISE[0], EMPRISE[2]),
            EMPRISE[1] - EMPRISE[0],
            EMPRISE[3] - EMPRISE[2],
            fill=False,
            edgecolor=ROUGE,
            linewidth=0.8,
            transform=proj,
            zorder=5,
        )
    )
    ins.spines["geo"].set_linewidth(0.5)

    cax = fig.add_axes([0.835, 7.0 / h_fig, 0.022, (h_carte - h_ins - 8.0) / h_fig])
    bornes = [CLASSES_BATHY[i][1] for i in visibles]
    etiquettes = [f"{a}–{b}" for a, b in zip(bornes, bornes[1:])] + [f"> {bornes[-1]}"]
    cmap = ListedColormap(bleus)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=BoundaryNorm(np.arange(len(bornes) + 1) - 0.5, cmap.N)
    )
    cb = fig.colorbar(sm, cax=cax, ticks=range(len(bornes)))
    cb.ax.set_yticklabels(etiquettes)
    cb.ax.invert_yaxis()
    cb.set_label("Depth (m)")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    ecrire(fig, "Figure_01_studyarea", apercu)
    plt.close(fig)
    return 0


DONNEES_ARTICLE = os.environ.get("ABASIN_DATA", os.path.join(RACINE, "donnees_article"))
ALLSAT_MARS = os.path.join(
    DONNEES_ARTICLE,
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_"
    "multi-vars_4.94W-9.94E_35.06N-40.94N_1993-01-01-2023-12-31.nc",
)
MDT = os.path.join(
    DONNEES_ARTICLE,
    "cmems_obs-sl_eur_phy-mdt_my_l4-0.0625deg_P20Y_mdt-v-u_"
    "4.97W-9.97E_35.03N-40.97N_2003-01-01.nc",
)
ESPACEMENT_FLECHES_DEG = 0.5
ECHELLE_FLECHES = 4.0
REFERENCE_FLECHE = 0.2


def moments_par_tranches(var, n, taille=366):
    s1 = s2 = c = None
    for d in range(0, n, taille):
        a = np.asarray(var.isel(time=slice(d, min(d + taille, n))).values, float)
        ok = np.isfinite(a)
        x = np.where(ok, a, 0.0)
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = (x * x).sum(0) if s2 is None else s2 + (x * x).sum(0)
        c = ok.sum(0) if c is None else c + ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(c > 0, s1 / np.maximum(c, 1), np.nan)
        v = np.where(c > 1, (s2 - c * m * m) / np.maximum(c - 1, 1), np.nan)
    return m, np.sqrt(np.maximum(v, 0.0))


def figure_2(apercu=None, alti=None, mdt=None):
    import cartopy.crs as ccrs
    import xarray as xr
    from matplotlib.colors import Normalize

    alti, mdt = alti or ALLSAT_MARS, mdt or MDT
    for f in (alti, mdt):
        if not os.path.isfile(f):
            return 2
    plt = style()
    with xr.open_dataset(alti) as d:
        manque = {"adt", "sla", "ugos", "vgos", "ugosa", "vgosa"} - set(d.data_vars)
        if manque:
            return 1
        n = d.sizes["time"]
        lat, lon = d["latitude"].values, d["longitude"].values
        adt, _ = moments_par_tranches(d["adt"], n)
        u, _ = moments_par_tranches(d["ugos"], n)
        v, _ = moments_par_tranches(d["vgos"], n)
        _, sla_et = moments_par_tranches(d["sla"], n)
        eke, _ = moments_par_tranches(
            0.5 * (d["ugosa"] ** 2 + d["vgosa"] ** 2) * 1e4, n
        )
    with xr.open_dataset(mdt) as d:
        mlat, mlon = d["latitude"].values, d["longitude"].values
        mdt2 = np.asarray(d["mdt"].squeeze().values, float)
        mu = np.asarray(d["u"].squeeze().values, float)
        mv = np.asarray(d["v"].squeeze().values, float)
    proj = ccrs.PlateCarree()
    vmin = float(min(np.nanmin(adt), np.nanmin(mdt2)))
    vmax = float(max(np.nanmax(adt), np.nanmax(mdt2)))
    hauteur = Normalize(vmin, vmax)
    fig = nouvelle_figure(plt, 112)
    gs = fig.add_gridspec(2, 2)
    panneaux = (
        ("a", lat, lon, adt, "viridis", hauteur, "Mean ADT (m)", (u, v)),
        ("b", lat, lon, eke, "magma_r", None, r"Mean EKE (cm$^2$ s$^{-2}$)", None),
        (
            "c",
            lat,
            lon,
            sla_et * 100.0,
            "magma_r",
            None,
            "SLA standard deviation (cm)",
            None,
        ),
        ("d", mlat, mlon, mdt2, "viridis", hauteur, "MDT (m)", (mu, mv)),
    )
    for i, (l, la, lo, champ, cmap, norm, lab, vit) in enumerate(panneaux):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection=proj)
        fond_de_carte(ax, ccrs, gauche=(i % 2 == 0))
        m = ax.pcolormesh(
            lo,
            la,
            champ,
            cmap=cmap,
            norm=norm,
            transform=proj,
            zorder=1,
            shading="nearest",
            rasterized=True,
        )
        if vit is not None:
            px, py = pas_des_fleches(lo, ESPACEMENT_FLECHES_DEG), pas_des_fleches(
                la, ESPACEMENT_FLECHES_DEG
            )
            q = ax.quiver(
                lo[::px],
                la[::py],
                vit[0][::py, ::px],
                vit[1][::py, ::px],
                color="white",
                scale=ECHELLE_FLECHES,
                scale_units="width",
                width=0.0022,
                headwidth=3.5,
                transform=proj,
                zorder=4,
            )
            ax.quiverkey(
                q,
                0.80,
                0.07,
                REFERENCE_FLECHE,
                rf"{REFERENCE_FLECHE:g} m s$^{{-1}}$",
                labelpos="E",
                coordinates="axes",
                color="black",
                labelcolor="black",
                fontproperties={"size": 6.5},
                zorder=7,
            )
        barre(fig, m, ax, lab)
        lettre(ax, l)
    ecrire(fig, "Figure_02_mean_circulation", apercu)
    plt.close(fig)
    return 0


TWOSAT = os.path.join(
    DOSSIER,
    "donnees",
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_"
    "4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc",
)
SERIE_EKE = os.path.join(DOSSIER, "donnees", "eke_bassin_twosat_mensuelle.csv")
CYCLE_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab14_cycle_saisonnier.csv"
)
ANOMALIE_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab15_bascule_de_l_anomalie.csv"
)
FENETRE_GLISSANTE = 12


def serie_eke_twosat():
    if os.path.isfile(SERIE_EKE):
        r = lire_csv(SERIE_EKE)
        return (
            np.array([int(x["annee"]) for x in r]),
            np.array([int(x["mois"]) for x in r]),
            np.array([float(x["eke_cm2_s2"]) for x in r]),
        )
    import xarray as xr

    with xr.open_dataset(TWOSAT) as d:
        eke = 0.5 * (d["ugosa"] ** 2 + d["vgosa"] ** 2) * 1e4
        s = eke.mean(("latitude", "longitude")).resample(time="1MS").mean()
        an, mo, v = s["time"].dt.year.values, s["time"].dt.month.values, s.values
    with io.open(SERIE_EKE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["annee", "mois", "eke_cm2_s2"])
        for a_, m_, x in zip(an, mo, v):
            w.writerow([int(a_), int(m_), f"{float(x):.6f}"])
    return an, mo, v


def climatologie(mois, v):
    moy = np.array([v[mois == k].mean() for k in range(1, 13)])
    et = np.array([v[mois == k].std(ddof=1) for k in range(1, 13)])
    return moy, et


def moyenne_glissante(v, n=FENETRE_GLISSANTE):
    out = np.full(v.size, np.nan)
    c = np.convolve(v, np.ones(n) / n, mode="valid")
    out[n // 2 : n // 2 + c.size] = c
    return out


def barres_signees(ax, t, y, alpha=0.6):
    ax.fill_between(
        t, 0, y, where=y >= 0, color=ROUGE, alpha=alpha, linewidth=0, step="mid"
    )
    ax.fill_between(
        t, 0, y, where=y < 0, color=BLEU, alpha=alpha, linewidth=0, step="mid"
    )
    ax.axhline(0, color="black", linewidth=0.5)


def figure_3(apercu=None):
    plt = style()
    an, mo, v = serie_eke_twosat()
    moy, et = climatologie(mo, v)
    anom = v - moy[mo - 1]
    t14 = {int(r["mois"]): r for r in lire_csv(CYCLE_THESE)}
    t15 = lire_csv(ANOMALIE_THESE)[0]
    ecarts = [
        k
        for k in range(1, 13)
        if abs(moy[k - 1] - float(t14[k]["moyenne"])) > 5e-4
        or abs(et[k - 1] - float(t14[k]["ecart_type"])) > 5e-4
    ]
    if ecarts or abs(anom.std(ddof=1) - float(t15["ecart_type"])) > 5e-4:
        return 1
    if (int(np.argmax(moy)) + 1, int(np.argmin(moy)) + 1) != (9, 3):
        return 1
    t = an + (mo - 0.5) / 12.0
    fig = nouvelle_figure(plt, 105)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(t, v, color="0.6", linewidth=0.5, label="monthly mean")
    ax.plot(
        t,
        moyenne_glissante(v),
        color="black",
        linewidth=1.1,
        label=f"{FENETRE_GLISSANTE}-month running mean",
    )
    ax.set_ylabel(r"EKE (cm$^2$ s$^{-2}$)")
    ax.set_xlim(1993, 2024)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, float(np.nanmax(v)) * 1.18)
    lettre(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(1, 13)
    ax.errorbar(
        x,
        moy,
        yerr=et,
        fmt="o-",
        color="black",
        ecolor="0.6",
        markersize=2.5,
        linewidth=0.9,
        elinewidth=0.7,
        capsize=1.5,
    )
    for k, c in ((int(np.argmax(moy)), ROUGE), (int(np.argmin(moy)), BLEU)):
        ax.plot(x[k], moy[k], "o", color=c, markersize=4, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MOIS])
    ax.set_xlim(0.4, 12.6)
    ax.set_ylabel(r"EKE (cm$^2$ s$^{-2}$)")
    lettre(ax, "b")
    ax = fig.add_subplot(gs[1, :])
    barres_signees(ax, t, anom)
    ax.plot(t, moyenne_glissante(anom), color="black", linewidth=1.0)
    ax.set_xlim(1993, 2024)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"EKE anomaly (cm$^2$ s$^{-2}$)")
    lettre(ax, "c")
    ecrire(fig, "Figure_03_EKE_timeseries", apercu)
    plt.close(fig)
    return 0


INTERVALLES_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab09_intervalle_par_blocs.csv"
)
ONDELETTES_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab22_ondelettes_significativite.csv"
)
BLOC_DU_TEXTE = 24
THESES = [
    os.path.join(RACINE, "..", "THESE_2026", "these-marge-algerienne"),
    os.path.join(RACINE, "these"),
    os.path.join(RACINE, "..", "these-marge-algerienne"),
    os.environ.get("THESE_DEPOT", ""),
]


def module_de_these(nom):
    for d in THESES:
        p = os.path.join(d, "src", "ch03_altimetrie", f"{nom}.py")
        if d and os.path.isfile(p):
            for q in (
                os.path.join(d, "src"),
                os.path.join(d, "src", "ch03_altimetrie"),
                d,
            ):
                if q not in sys.path:
                    sys.path.insert(0, q)
            import importlib

            return importlib.import_module(nom)
    return None


def module_ondelettes():
    return module_de_these("ch03_tab22_ondelettes")


def figure_4(apercu=None):
    plt = style()
    w = module_ondelettes()
    if w is None:
        return 2
    an, mo, v = serie_eke_twosat()
    moy, _ = climatologie(mo, v)
    anom = v - moy[mo - 1]
    res = w.analyser(anom)
    res115 = w.analyser(anom, alpha=w.ALPHA_MANUSCRIT)
    publie = {
        r["periode_annoncee_mois"]: r
        for r in lire_csv(ONDELETTES_THESE)
        if r["fond_rouge"].startswith("alpha mesure")
    }
    for L in w.rapport(res):
        ref = float(publie[L["periode_annoncee_mois"]]["rapport_au_seuil"])
        if abs(L["rapport_au_seuil"] - ref) > 1e-5 or L["franchit"] != "NON":
            return 1
    puis, lam, sc = w._ondelettes(anom)
    _, _, s_loc, _ = w._seuils(anom, lam, sc)
    coi = w._coi(len(anom), sc, lam)
    t = an + (mo - 0.5) / 12.0
    iv = lire_csv(INTERVALLES_THESE)
    retenu = bloc_retenu(
        iv, module_de_these("ch03_tab09_incertitude_pente").bloc_retenu
    )
    if retenu != BLOC_DU_TEXTE:
        return 1
    L = np.array([int(r["bloc_mois"]) for r in iv])
    pe = np.array([float(r["pente_cm2_s2_par_decennie"]) for r in iv])
    lo = np.array([float(r["borne_basse_corrigee"]) for r in iv])
    hi = np.array([float(r["borne_haute_corrigee"]) for r in iv])
    if not ((lo < 0) & (hi > 0)).all():
        return 1

    fig = nouvelle_figure(plt, 118)
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0])
    ax = fig.add_subplot(gs[0, :])
    for i in range(L.size):
        c = "black" if L[i] == retenu else "0.6"
        ax.errorbar(
            [i],
            [pe[i]],
            yerr=[[pe[i] - lo[i]], [hi[i] - pe[i]]],
            fmt="o",
            color=c,
            ecolor=c,
            capsize=2,
            markersize=3,
            elinewidth=0.9,
        )
    ax.axhline(0, color=GRIS_SEUIL, linewidth=0.7, linestyle="--")
    ax.set_xticks(np.arange(L.size))
    ax.set_xticklabels([str(x) for x in L])
    ax.set_xlim(-0.6, L.size - 0.4)
    ax.set_xlabel("Block length of the moving-block bootstrap (months)")
    ax.set_ylabel("Sen's slope\n" + r"(cm$^2$ s$^{-2}$ decade$^{-1}$)")
    lettre(ax, "a")

    ax = fig.add_subplot(gs[1, :2])
    norm = puis / anom.var(ddof=1)
    cs = ax.contourf(
        t,
        lam,
        np.log2(norm),
        levels=np.arange(-4, 4.5, 0.5),
        cmap="viridis",
        extend="both",
    )
    for c in cs.collections if hasattr(cs, "collections") else [cs]:
        c.set_rasterized(True)
    ax.contour(
        t, lam, puis / s_loc[:, None], levels=[1.0], colors="white", linewidths=0.7
    )
    ax.fill_between(
        t, coi, lam.max(), facecolor="white", alpha=0.45, edgecolor="none", zorder=2
    )
    ax.fill_between(
        t,
        coi,
        lam.max(),
        facecolor="none",
        hatch="////",
        edgecolor="0.35",
        linewidth=0,
        zorder=3,
    )
    ax.plot(t, coi, color="0.2", linewidth=0.6, zorder=4)
    ax.set_yscale("log", base=2)
    ax.set_ylim(w.PERIODES_ARTICLE[1], w.PERIODES_ARTICLE[0])
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.minorticks_off()
    ax.set_xlim(1993, 2024)
    ax.set_xlabel("Year")
    ax.set_ylabel("Period (months)")
    cax = ax.inset_axes([1.015, 0.0, 0.022, 1.0])
    cb = fig.colorbar(cs, cax=cax)
    cb.set_label(r"log$_2$(power / variance)")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2)
    cb.set_ticks(range(-4, 5, 2))
    lettre(ax, "b")

    ax = fig.add_subplot(gs[1, 2])
    ok = res["dans"]
    ax.plot(
        res["global_spec"][ok],
        res["lam"][ok],
        color="black",
        linewidth=1.1,
        label="global spectrum",
    )
    ax.plot(
        res["seuil"][ok],
        res["lam"][ok],
        color=ROUGE,
        linewidth=0.9,
        label=r"95 %, measured $\alpha$",
    )
    ax.plot(
        res115["seuil"][ok],
        res115["lam"][ok],
        color="0.5",
        linewidth=0.9,
        linestyle="--",
        label=r"95 %, submitted $\alpha$",
    )
    ax.set_yscale("log", base=2)
    ax.set_ylim(w.PERIODES_ARTICLE[1], w.PERIODES_ARTICLE[0])
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.minorticks_off()
    ax.set_xlim(0, None)
    ax.set_xlabel(r"Power (cm$^4$ s$^{-4}$)")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.3, borderaxespad=0.2)
    lettre(ax, "c")
    ecrire(fig, "Figure_04_MannKendall_Wavelet", apercu)
    plt.close(fig)
    return 0


PARTS_EOF = os.path.join(DOSSIER, "sorties", "figure_05_variance_shares.csv")
PARTS_TEXTE = {"tel quel": (54.2, 5.0, 3.6, 3.4), "tendance retiree": (35.2,)}


def figure_5(apercu=None):
    import cartopy.crs as ccrs
    import xarray as xr
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    sys.path.insert(0, DOSSIER)
    import eof_products as c13

    plt = style()
    with xr.open_dataset(c13.ALLSAT) as ds:
        valide = np.isfinite(ds["sla"].isel(time=0).values)
        lats, lons = ds["latitude"].values, ds["longitude"].values
    mois, X = c13.serie_sla(c13.ALLSAT, valide)
    m = np.array([k for _, k in mois])
    A = c13.anomalie_mensuelle(X, m)
    r = c13.eof(A)
    r_d = c13.eof(c13.retirer_tendance(A))
    lignes = []
    for nom, rr in (("tel quel", r), ("tendance retiree", r_d)):
        for k, p in enumerate(rr["parts"], start=1):
            lignes.append(
                {"champ": nom, "mode": f"EOF{k}", "part_de_variance": f"{p:.6f}"}
            )
    for nom, rr in (("tel quel", r), ("tendance retiree", r_d)):
        for k, attendu in enumerate(PARTS_TEXTE[nom]):
            if round(100 * rr["parts"][k], 1) != attendu:
                return 1
    an = np.array([a for a, _ in mois])
    t = an + (m - 0.5) / 12.0
    proj = ccrs.PlateCarree()
    fig = nouvelle_figure(plt, 172)
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.0])
    for k in range(4):
        pc = r["pcs"][k]
        pcn = (pc - pc.mean()) / pc.std(ddof=1)
        cov = (A - A.mean(axis=0)).T @ pcn / (len(pcn) - 1) * 100.0
        champ = np.full(valide.shape, np.nan)
        champ[valide] = cov
        if k == 0:
            signe = np.sign(cov[int(np.argmax(np.abs(cov)))])
            part = float(np.mean(np.sign(cov) == signe))
            lignes.append(
                {
                    "champ": "tel quel",
                    "mode": "EOF1 mailles du signe de l'extremum",
                    "part_de_variance": f"{part:.6f}",
                }
            )
        ax = fig.add_subplot(gs[k, 0], projection=proj)
        fond_de_carte(ax, ccrs, bas=(k == 3))
        lim = float(np.nanmax(np.abs(champ)))
        mm = ax.pcolormesh(
            lons,
            lats,
            champ,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0.0, -lim, lim),
            transform=proj,
            zorder=1,
            shading="nearest",
            rasterized=True,
        )
        ax.contour(
            lons,
            lats,
            champ,
            levels=[0.0],
            colors="black",
            linewidths=0.5,
            transform=proj,
            zorder=2,
        )
        cb = barre(fig, mm, ax, "Covariance (cm)", lieu="right", aspect=10)
        cb.locator = MaxNLocator(nbins=4, symmetric=True)
        cb.formatter = FuncFormatter(lambda x, _: f"{x:g}".replace("-", "\u2212"))
        cb.update_ticks()
        lettre(ax, "aceg"[k])
        ax = fig.add_subplot(gs[k, 1])
        barres_signees(ax, t, pcn, alpha=0.45)
        ax.plot(t, moyenne_glissante(pcn), color="black", linewidth=0.9)
        ax.set_xlim(1993, 2024)
        ax.set_ylim(-borne_symetrique(pcn), borne_symetrique(pcn))
        ax.set_ylabel(f"PC{k + 1} (standardised)")
        if k == 3:
            ax.set_xlabel("Year")
        else:
            ax.tick_params(labelbottom=False)
        lettre(ax, "bdfh"[k])
    ecrire(fig, "Figure_05_EOF_SLA", apercu)
    plt.close(fig)
    with io.open(PARTS_EOF, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)
    return 0


DERIVE_ZONES = os.path.join(DOSSIER, "donnees", "zones_champs_moyens.npz")
STABILITE = os.path.join(DOSSIER, "sorties", "zonation_stability.csv")
SENSIBILITE = os.path.join(DOSSIER, "sorties", "zone_numerical_sensitivity.csv")
DESCRIPTION = os.path.join(DOSSIER, "sorties", "zone_description.csv")
COULEURS_ZONES = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#CC79A7"]
QUANTILE_DIVERGENCE = 98.0


def figure_6(apercu=None):
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

    sys.path.insert(0, DOSSIER)
    from zonation import divergence_masquee, metrique_du_pas

    plt = style()
    z = np.load(DERIVE_ZONES, allow_pickle=False)
    lats, lons, e = z["lats"], z["lons"], z["etiquettes"].astype(int)
    u, v = z["u"].astype(float), z["v"].astype(float)
    dx, dy = metrique_du_pas(lons, lats)
    div = divergence_masquee(u, v, dx, dy) * 1e6
    desc = lire_csv(DESCRIPTION)
    tirages = lire_csv(STABILITE)
    sens = {r["grandeur"]: r["valeur"] for r in lire_csv(SENSIBILITE)}
    ch_fig = [float(sens[f"critere_k{k}_chemin_figure"]) for k in range(3, 7)]
    if int(e.max()) != 6 or not all(a < b for a, b in zip(ch_fig, ch_fig[1:])):
        return 1

    proj = ccrs.PlateCarree()
    fig = nouvelle_figure(plt, 112)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.0])
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    fond_de_carte(ax, ccrs)
    cmap = ListedColormap(COULEURS_ZONES)
    m = ax.pcolormesh(
        lons,
        lats,
        np.where(e > 0, e, np.nan),
        cmap=cmap,
        norm=BoundaryNorm(np.arange(0.5, 7.5, 1), 6),
        transform=proj,
        zorder=1,
        shading="nearest",
        rasterized=True,
    )
    cb = barre(
        fig,
        m,
        ax,
        "Class of the mean velocity field (not a contiguous region)",
        ticks=range(1, 7),
    )
    cb.ax.tick_params(length=0)
    lettre(ax, "a")

    ax = fig.add_subplot(gs[0, 1], projection=proj)
    fond_de_carte(ax, ccrs, gauche=False)
    lim = float(np.nanpercentile(np.abs(div), QUANTILE_DIVERGENCE))
    m = ax.pcolormesh(
        lons,
        lats,
        div,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(0.0, -lim, lim),
        transform=proj,
        zorder=1,
        shading="nearest",
        rasterized=True,
    )
    barre(fig, m, ax, r"Mean horizontal divergence ($10^{-6}$ s$^{-1}$)", extend="both")
    lettre(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    ks = np.arange(3, 7)
    for i, r in enumerate(t for t in tirages if t["nature"] == "bruit borne"):
        ax.plot(
            ks,
            [float(r[f"critere_k{k}"]) for k in ks],
            color="0.7",
            linewidth=0.5,
            label="perturbed fields" if i == 0 else None,
        )
    ch64 = [float(sens[f"critere_k{k}_chemin_float64"]) for k in ks]
    ax.plot(
        ks,
        ch64,
        "s--",
        color=BLEU,
        markersize=3,
        linewidth=0.9,
        label="mean computed in double precision",
    )
    ax.plot(
        ks,
        ch_fig,
        "o-",
        color="black",
        markersize=3,
        linewidth=1.0,
        label="mean as in the submitted analysis",
    )
    ax.set_xticks(ks)
    ax.set_xlabel(r"Number of classes $\mathit{N}$")
    ax.set_ylabel("Calinski–Harabasz criterion")
    ax.legend(loc="lower right")
    lettre(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    zz = np.array([int(r["zone"]) for r in desc])
    larg = 0.27
    for i, (col, lab, c) in enumerate(
        (
            ("u_cm_s", r"$\mathit{U}$", "0.35"),
            ("v_cm_s", r"$\mathit{V}$", "0.7"),
            ("vitesse_cm_s", "speed", "black"),
        )
    ):
        ax.bar(
            zz + (i - 1) * larg,
            [float(r[col]) for r in desc],
            width=larg,
            color=c,
            label=lab,
        )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(zz)
    ax.set_xlabel("Class")
    ax.set_ylabel(r"Velocity (cm s$^{-1}$)")
    for t, c in zip(ax.get_xticklabels(), COULEURS_ZONES):
        t.set_bbox(dict(facecolor=c, edgecolor="none", pad=1.2))
    ax.tick_params(axis="x", length=0, pad=4)
    ax.legend(loc="upper right", ncol=3)
    haut = max(float(r["vitesse_cm_s"]) for r in desc)
    ax.set_ylim(None, haut * 1.25)
    lettre(ax, "d")
    ecrire(fig, "Figure_06_CAH_zonation", apercu)
    plt.close(fig)
    return 0


TEXTE_FIG7 = {
    "mois_max": {"anticyclone": "August", "cyclone": "September"},
    "extremes": {
        "anticyclone": (2003, 108, 2023, 313),
        "cyclone": (2002, 46, 2012, 229),
    },
}


def figure_7(apercu=None):
    import cartopy.crs as ccrs
    from matplotlib.colors import LogNorm

    plt = style()
    a = {k: lire_atlas(p) for k, p in ATLAS.items()}
    n_annees = 31
    defauts = []
    for k, d in a.items():
        cyc = cycle_saisonnier(d["mois"], n_annees)
        mois = MOIS_EN[int(np.argmax(cyc))]
        if mois != TEXTE_FIG7["mois_max"][k]:
            defauts.append(
                f"{k} : maximum saisonnier en {mois}, le texte dit "
                f"{TEXTE_FIG7['mois_max'][k]}"
            )
        ans, n = comptes_annuels(d["annee"])
        e = extremes(ans, n)
        if e[:4] != TEXTE_FIG7["extremes"][k] or e[4]:
            defauts.append(
                f"{k} : extremes {e}, le texte dit " f"{TEXTE_FIG7['extremes'][k]}"
            )
    if defauts:
        return 1

    fig = nouvelle_figure(plt, 150)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.80, 0.85, 0.85])
    proj = ccrs.PlateCarree()
    gx = int(round((EMPRISE[1] - EMPRISE[0]) / LARGEUR_HEXAGONE_DEG))
    vmax = 1
    for k in a:
        h = (
            plt.figure()
            .add_subplot()
            .hexbin(a[k]["lon"], a[k]["lat"], gridsize=gx, extent=EMPRISE)
        )
        vmax = max(vmax, h.get_array().max())
        plt.close(h.figure)
    cartes = []
    for i, k in enumerate(a):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        fond_de_carte(ax, ccrs, gauche=(i == 0))
        hb = ax.hexbin(
            a[k]["lon"],
            a[k]["lat"],
            gridsize=gx,
            extent=EMPRISE,
            mincnt=1,
            norm=LogNorm(vmin=1, vmax=vmax),
            cmap="viridis",
            linewidths=0,
            transform=proj,
            zorder=1,
            rasterized=True,
        )
        ax.text(
            0.985,
            0.04,
            NOM[k],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            zorder=10,
            color="0.1",
        )
        lettre(ax, "ab"[i])
        cartes.append(ax)
    cb = barre(fig, hb, cartes, "Documented days per hexagon", shrink=0.6, aspect=45)
    cb.set_ticks([1, 10, 100, 1000])
    cb.set_ticklabels(["1", "10", "100", "1000"])
    cb.ax.minorticks_off()

    for j, (var, xl, unite, bins) in enumerate(
        (
            ("amplitude_cm", "Amplitude (cm)", "cm", np.arange(0, 15.25, 0.25)),
            ("rayon_km", "Speed radius (km)", "km", np.arange(0, 61, 1)),
        )
    ):
        ax = fig.add_subplot(gs[1, j])
        for k in a:
            v = a[k][var]
            ax.hist(
                v,
                bins=bins,
                density=True,
                histtype="step",
                color=COULEUR[k],
                linewidth=0.9,
                label=f"{NOM[k]} (mean {v.mean():.1f} {unite})",
            )
            ax.axvline(v.mean(), color=COULEUR[k], linewidth=0.7, linestyle="--")
        ax.set_xlabel(xl)
        ax.set_ylabel("Probability density")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, None)
        ax.legend(loc="upper right")
        lettre(ax, "cd"[j])

    ax = fig.add_subplot(gs[2, 0])
    larg = 0.4
    for i, k in enumerate(a):
        ans, n = comptes_annuels(a[k]["annee"])
        ax.bar(
            ans + (i - 0.5) * larg,
            n,
            width=larg,
            color=COULEUR[k],
            label=NOM[k],
            linewidth=0,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Documented days per year")
    ax.set_xlim(1992.3, 2023.7)
    ax.set_ylim(0, 365)
    ax.legend(loc="upper left", ncol=2)
    lettre(ax, "e")

    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(1, 13)
    haut = 0.0
    for i, k in enumerate(a):
        cyc = cycle_saisonnier(a[k]["mois"], n_annees)
        haut = max(haut, cyc.max())
        bord = ["black" if m == np.argmax(cyc) else COULEUR[k] for m in range(12)]
        ep = [0.9 if m == np.argmax(cyc) else 0.0 for m in range(12)]
        ax.bar(
            x + (i - 0.5) * larg,
            cyc,
            width=larg,
            color=COULEUR[k],
            edgecolor=bord,
            linewidth=ep,
            label=NOM[k],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in MOIS])
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(0, haut * 1.22)
    ax.set_ylabel("Documented days per month")
    ax.legend(loc="upper left", ncol=2)
    lettre(ax, "f")

    ecrire(fig, "Figure_07_Eddy_Atlas", apercu)
    plt.close(fig)
    return 0


WEMO_ORIGINE = os.path.join(RACINE, "origine_2026-03", "wemo_index_era5_1993_2023.csv")
BALAYAGE_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab20_balayage_des_retards.csv"
)
VERDICT_THESE = os.path.join(
    DOSSIER, "sorties_these", "these_ch03_tab21_verdict_et_effectif.csv"
)
SERIE_8 = "reconstruction d'origine (signe inverse)"
SIGNE_8 = -1.0
ENERGIES_8 = (
    ("turbulente_quotidienne", "EKE (Eq. 1)", BLEU),
    ("totale_quotidienne", "kinetic energy of the total flow", ROUGE),
)


def figure_8(apercu=None):
    plt = style()
    an, mo, v = serie_eke_twosat()
    moy, _ = climatologie(mo, v)
    a = v - moy[mo - 1]
    wemo = SIGNE_8 * np.array([float(r["wemo_era5"]) for r in lire_csv(WEMO_ORIGINE)])
    verd = {r["energie"]: r for r in lire_csv(VERDICT_THESE) if r["serie"] == SERIE_8}
    r0 = float(np.corrcoef(wemo, a)[0, 1])
    attendu = SIGNE_8 * float(verd["turbulente_quotidienne"]["r_au_retard_zero"])
    if abs(r0 - attendu) > 1e-5:
        return 1
    bal = [r for r in lire_csv(BALAYAGE_THESE) if r["serie"] == SERIE_8]
    t = an + (mo - 0.5) / 12.0
    fig = nouvelle_figure(plt, 118)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.0])
    ax = fig.add_subplot(gs[0])
    ax.plot(
        t, wemo, color="0.25", linewidth=0.6, label="WeMO index (ERA5 reconstruction)"
    )
    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_ylabel("WeMO index (standardised)")
    ax.set_xlim(1993, 2024)
    ax.set_ylim(-borne_symetrique(wemo, 1.15), borne_symetrique(wemo, 1.15))
    ax2 = ax.twinx()
    ax2.plot(t, a, color=BLEU, linewidth=0.6, alpha=0.9, label="EKE anomaly (TWO-SAT)")
    ax2.set_ylim(-borne_symetrique(a, 1.15), borne_symetrique(a, 1.15))
    ax2.set_ylabel(r"EKE anomaly (cm$^2$ s$^{-2}$)", color=BLEU)
    ax2.tick_params(axis="y", colors=BLEU, width=0.5)
    ax2.spines["right"].set_color(BLEU)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=2,
        borderaxespad=0.1,
    )
    lettre(ax, "a")
    ax = fig.add_subplot(gs[1])
    for e, lab, c in ENERGIES_8:
        pts = sorted(
            (int(r["retard_mois"]), SIGNE_8 * float(r["r"]))
            for r in bal
            if r["energie"] == e
        )
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o-",
            color=c,
            markersize=2.2,
            linewidth=0.9,
            label=lab,
        )
        sb = float(verd[e]["seuil_bonferroni"])
        sn = float(verd[e]["seuil_nominal"])
        for y in (sb, -sb):
            ax.axhline(y, color=GRIS_SEUIL, linewidth=0.7, linestyle="--")
        for y in (sn, -sn):
            ax.axhline(y, color=GRIS_SEUIL, linewidth=0.5, linestyle=":")
    ax.plot(
        [],
        [],
        color=GRIS_SEUIL,
        linestyle="--",
        linewidth=0.7,
        label="threshold corrected for lag scanning",
    )
    ax.plot(
        [],
        [],
        color=GRIS_SEUIL,
        linestyle=":",
        linewidth=0.5,
        label="nominal threshold (not a test)",
    )
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Lag (months); positive lag: the WeMO leads")
    ax.set_ylabel("Correlation coefficient")
    ax.set_xlim(-25, 25)
    lim = max(abs(float(r["r"])) for r in bal) * 1.0
    ax.set_ylim(-max(lim, 0.25) * 1.15, max(lim, 0.25) * 1.45)
    ax.legend(loc="upper left", ncol=2)
    lettre(ax, "b")
    ecrire(fig, "Figure_08_WeMO_EKE_1993_2023", apercu)
    plt.close(fig)
    return 0


SERIE_SYNTHESE = os.path.join(
    DOSSIER, "donnees", "eke_mensuelle_trois_champs_1993_2023.csv"
)
TENDANCES = os.path.join(DOSSIER, "sorties", "products_trends.csv")
NORTH = os.path.join(DOSSIER, "sorties", "eof_north.csv")
BANDES = os.path.join(DOSSIER, "sorties", "ekman_africa_index_by_band.csv")
ETE, HIVER = "#E69F00", "#0072B2"


def figure_9(apercu=None):
    sys.path.insert(0, DOSSIER)
    from synthesis import annuelle, contigues, retards, separabilite

    plt = style()
    sources = (
        SERIE_SYNTHESE,
        TENDANCES,
        PARTS_EOF,
        NORTH,
        BALAYAGE_THESE,
        VERDICT_THESE,
        BANDES,
    )
    manquants = [p for p in sources if not os.path.isfile(p)]
    if manquants:
        return 1
    serie, tend = lire_csv(SERIE_SYNTHESE), {l["champ"]: l for l in lire_csv(TENDANCES)}
    parts = lire_csv(PARTS_EOF)
    north = lire_csv(NORTH)
    bal = [r for r in lire_csv(BALAYAGE_THESE) if r["serie"] == SERIE_8]
    verd = {r["energie"]: r for r in lire_csv(VERDICT_THESE) if r["serie"] == SERIE_8}
    bandes = lire_csv(BANDES)

    fig = nouvelle_figure(plt, 120)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    for champ, nom, c in (
        ("allsat_025", "ALL-SAT block-averaged to 0.25°", "0.45"),
        ("twosat_025", "TWO-SAT (0.25°)", BLEU),
    ):
        ans, val = annuelle(serie, champ)
        l = tend[champ]
        verdict = "includes zero" if l["zero_dedans"] == "oui" else "excludes zero"
        ax.plot(
            ans,
            [x * 1e4 for x in val],
            "o-",
            color=c,
            markersize=2.2,
            linewidth=1.0,
            label=f"{nom}: {100 * float(l['pente_relative_par_decennie']):+.1f} % "
            f"per decade, interval {verdict}",
        )
    ax.set_xlim(1992.5, 2023.5)
    ax.set_ylabel(r"Annual mean EKE (cm$^2$ s$^{-2}$)")
    ax.set_xlabel("Year")
    bas, haut = ax.get_ylim()
    ax.set_ylim(bas, haut + 0.32 * (haut - bas))
    ax.legend(loc="upper left", fontsize=6)
    lettre(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    p = {(l["champ"], l["mode"]): 100 * float(l["part_de_variance"]) for l in parts}
    modes = ["EOF1", "EOF2", "EOF3", "EOF4"]
    x = np.arange(4)
    tq = [p[("tel quel", m)] for m in modes]
    dt = [p[("tendance retiree", m)] for m in modes]
    ax.bar(x - 0.19, tq, 0.38, color="0.45", label="as computed")
    ax.bar(x + 0.19, dt, 0.38, color=BLEU, label="detrended")
    haut = max(max(tq), max(dt))
    ax.set_ylim(0, haut * 1.45)
    seps = [
        (nom, {pp: ok for pp, _, _, ok in separabilite(north, champ)})
        for champ, nom in (("allsat_025", "ALL-SAT"), ("twosat_025", "TWO-SAT"))
    ]
    ax.text(
        -0.42,
        haut * 1.37,
        "Separable from the next mode (North):",
        ha="left",
        va="center",
        fontsize=6,
    )
    for k, (nom, s_) in enumerate(seps):
        y = haut * (1.25 - 0.10 * k)
        ax.text(-0.42, y, nom, ha="left", va="center", fontsize=6)
        for i in range(3):
            ok = s_.get(f"{modes[i]}/{modes[i + 1]}")
            ax.text(
                i + 0.5,
                y,
                "yes" if ok else "no",
                ha="center",
                va="center",
                fontsize=6,
                color="black" if ok else "0.5",
                fontweight="bold" if ok else "normal",
            )
    ax.axhline(haut * 1.07, color="0.75", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Variance explained (%)")
    ax.legend(loc="center right")
    lettre(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    for e, lab, c in ENERGIES_8:
        pts = retards(bal, e, SERIE_8)
        ax.plot(
            [q[0] for q in pts],
            [SIGNE_8 * q[1] for q in pts],
            "o-",
            color=c,
            markersize=1.8,
            linewidth=0.9,
            label=lab,
        )
    sb, sn = float(verd["totale_quotidienne"]["seuil_bonferroni"]), float(
        verd["totale_quotidienne"]["seuil_nominal"]
    )
    for s, st, w in ((sb, "--", 0.7), (sn, ":", 0.5)):
        for sg in (1, -1):
            ax.axhline(sg * s, color=GRIS_SEUIL, linestyle=st, linewidth=w)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlim(-25, 25)
    ax.set_ylim(-0.26, 0.36)
    ax.set_xlabel("Lag (months); positive lag: the WeMO leads")
    ax.set_ylabel("Correlation coefficient")
    ax.legend(loc="upper left", fontsize=6)
    lettre(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    lon = np.array([(float(l["lon_min"]) + float(l["lon_max"])) / 2 for l in bandes])
    ete = np.array([float(l["ete"]) for l in bandes])
    hiv = np.array([float(l["hiver"]) for l in bandes])
    deb, lg = contigues(ete, positif=True)
    if lg:
        ax.axvspan(
            lon[deb] - 0.5,
            lon[deb + lg - 1] + 0.5,
            color=ETE,
            alpha=0.13,
            linewidth=0,
            zorder=0,
        )
    ax.bar(lon - 0.21, ete, 0.42, color=ETE, label="summer")
    ax.bar(lon + 0.21, hiv, 0.42, color=HIVER, label="winter")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlim(lon.min() - 0.6, lon.max() + 0.6)
    ax.set_xticks(range(-4, 11, 2))
    ax.set_xticklabels(
        [
            f"{abs(k)}°{'W' if k < 0 else ('E' if k > 0 else '')}"
            for k in range(-4, 11, 2)
        ]
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel(r"Offshore Ekman transport (m$^2$ s$^{-1}$)")
    ax.legend(loc="lower left")
    lettre(ax, "d")
    ecrire(fig, "Figure_09_synthesis", apercu)
    plt.close(fig)
    return 0


FIGURES = {
    "1": figure_1,
    "2": figure_2,
    "3": figure_3,
    "4": figure_4,
    "5": figure_5,
    "6": figure_6,
    "7": figure_7,
    "8": figure_8,
    "9": figure_9,
}


if __name__ == "__main__":
    apercu = None
    if "--apercu" in sys.argv:
        apercu = sys.argv[sys.argv.index("--apercu") + 1]
    if "--figure" in sys.argv:
        n = sys.argv[sys.argv.index("--figure") + 1]
        sys.exit(FIGURES[n](apercu))
    if "--toutes" in sys.argv:
        codes = {}
        for n, f in FIGURES.items():
            codes[n] = f(apercu)
        sys.exit(max(codes.values()))
