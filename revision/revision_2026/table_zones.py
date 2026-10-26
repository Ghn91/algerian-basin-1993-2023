"""LaTeX tables of the six classes of the zonation, generated from
zone_description.csv.

Usage: python table_zones.py --construire
"""

import csv
import io
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(DOSSIER, "sorties", "zone_description.csv")
CIBLE = os.path.join(DOSSIER, "manuscript", "tableau_des_zones.tex")

COLONNES = [
    "zone",
    "aire_km2",
    "lon_mediane",
    "lat_mediane",
    "u_cm_s",
    "v_cm_s",
    "vitesse_cm_s",
    "vorticite_sur_f_moyenne",
    "vorticite_part_cyclonique",
    "divergence_masquee_s-1",
    "eke_twosat_cm2_s-2_moyenne",
    "eke_twosat_cm2_s-2_couverture",
    "sst_degC_moyenne",
    "sst_degC_couverture",
    "chl_mg_m-3_moyenne",
    "chl_mg_m-3_couverture",
    "profondeur_m_mediane",
    "profondeur_m_couverture",
    "detections_anticyclone_par_1e4_km2",
    "detections_cyclone_par_1e4_km2",
]
COUVERTURES_DECLAREES = {"profondeur_m_couverture", "sst_degC_couverture"}


def f(x, d):
    return f"{float(x):.{d}f}".replace("-", "$-$")


def defauts(lignes):
    d = []
    if not lignes:
        return ["table vide"]
    manque = [c for c in COLONNES if c not in lignes[0]]
    if manque:
        d.append(f"colonnes absentes : {manque}")
        return d
    for l in lignes:
        for c in COLONNES:
            if (
                c.endswith("_couverture")
                and c not in COUVERTURES_DECLAREES
                and float(l[c]) < 1.0
            ):
                d.append(f"zone {l['zone']} : {c} = {l[c]}, non declaree")
    return d


def tableaux(lignes):
    sst_min = min(float(l["sst_degC_couverture"]) for l in lignes)
    a = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Kinematics of the six classes of the zonation "
        r"(Sect.~\ref{sec:zonation}), computed from the 1993--2023 mean MEDSEA "
        r"surface velocity field on which the classification is built. Area "
        r"in $10^{3}$~km$^{2}$; median position of the cells of each class; "
        r"mean zonal ($U$) and meridional ($V$) velocity and mean speed, in "
        r"cm~s$^{-1}$; mean relative vorticity of the mean field divided by "
        r"the local Coriolis parameter ($\zeta/f$, in $10^{-3}$), and the "
        r"fraction of cells where it is positive (cyclonic); mean horizontal "
        r"divergence, computed without differencing against land, in "
        r"$10^{-8}$~s$^{-1}$. The classes are not contiguous regions.}",
        r"\label{tab:zones-kinematics}",
        r"\begin{tabular}{crrrrrrrrr}",
        r"\toprule",
        r"Zone & Area & Lon. & Lat. & $U$ & $V$ & Speed & $\zeta/f$ & "
        r"Cycl. & Div. \\",
        r"\midrule",
    ]
    for l in lignes:
        a.append(
            " & ".join(
                [
                    l["zone"],
                    f(float(l["aire_km2"]) / 1e3, 1),
                    f(l["lon_mediane"], 2),
                    f(l["lat_mediane"], 2),
                    f(l["u_cm_s"], 1),
                    f(l["v_cm_s"], 1),
                    f(l["vitesse_cm_s"], 1),
                    f(float(l["vorticite_sur_f_moyenne"]) * 1e3, 1),
                    f(l["vorticite_part_cyclonique"], 3),
                    f(float(l["divergence_masquee_s-1"]) * 1e8, 2),
                ]
            )
            + r" \\"
        )
    a += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    b = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Properties associated with the six classes, none of which "
        r"enters the classification. EKE: 1993--2023 mean of the TWO-SAT "
        r"product (cm$^{2}$~s$^{-2}$), nearest 0.25\textdegree{} cell. SST: "
        r"1993--2023 mean of the L4 reprocessed product (\textdegree C), "
        r"nearest 0.05\textdegree{} cell, available on at least "
        + f(sst_min, 3)
        + r" of the cells of every class. Chl: 1999--2023 "
        r"mean of monthly means of surface chlorophyll-$a$ from the "
        r"biogeochemical reanalysis (mg~m$^{-3}$), a model coupled to the "
        r"physical reanalysis that defines the classes, hence descriptive "
        r"only. Depth: median depth (m) over the fraction of the class "
        r"covered by the bathymetric grid used here, given in parentheses. "
        r"AE, CE: daily anticyclonic and cyclonic detections of the eddy "
        r"catalogue per $10^{4}$~km$^{2}$ of class area over 1993--2023; "
        r"detections without tracking, so these are not generation sites.}",
        r"\label{tab:zones-properties}",
        r"\begin{tabular}{crrrrrr}",
        r"\toprule",
        r"Zone & EKE & SST & Chl & Depth & AE & CE \\",
        r"\midrule",
    ]
    for l in lignes:
        b.append(
            " & ".join(
                [
                    l["zone"],
                    f(l["eke_twosat_cm2_s-2_moyenne"], 1),
                    f(l["sst_degC_moyenne"], 2),
                    f(l["chl_mg_m-3_moyenne"], 3),
                    f(l["profondeur_m_mediane"], 0)
                    + " ("
                    + f(l["profondeur_m_couverture"], 2)
                    + ")",
                    f(l["detections_anticyclone_par_1e4_km2"], 1),
                    f(l["detections_cyclone_par_1e4_km2"], 1),
                ]
            )
            + r" \\"
        )
    b += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(a + b)


def lire(chemin=SOURCE):
    with io.open(chemin, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def construire():
    if not os.path.isfile(SOURCE):
        return 2
    lignes = lire()
    d = defauts(lignes)
    if d:
        return 1
    io.open(CIBLE, "w", encoding="utf-8", newline="\n").write(tableaux(lignes))
    return 0


if __name__ == "__main__":
    if "--construire" in sys.argv:
        sys.exit(construire())
    print(__doc__)
