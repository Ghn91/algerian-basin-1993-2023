"""Monthly volume transport through the Gibraltar-Ceuta section of the MEDSEA
physical reanalysis (eastward inflow and net), deseasonalised and
cross-correlated over lags -24 to +24 months with the reconstructed WeMO index
and with total-flow kinetic energy, against Bonferroni thresholds computed on
the effective sample size. The code first reproduces the WeMO correlation
stated in the manuscript at +1 month, and stops otherwise. A second mode sweeps the western
limit of the area over which the kinetic energy is averaged.

Usage: python gibraltar_inflow.py --telecharger | --produire |
       --deriver-bandes | --longitudes
"""

import csv
import io
import os
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(DOSSIER)
sys.path.insert(0, DOSSIER)
from nao_eke import correlation_au_retard, desaisonnaliser, n_effectif, seuils

PRODUIT = "MEDSEA_MULTIYEAR_PHY_006_004"
JEU = "cmems_mod_med_phy-cur_my_4.2km_P1M-m"
SECTION_LON = -5.35
EMPRISE = dict(
    minimum_longitude=-5.45,
    maximum_longitude=-5.25,
    minimum_latitude=35.80,
    maximum_latitude=36.20,
    minimum_depth=1.0182366371154785,
    maximum_depth=971.077880859375,
    start_datetime="1993-01-01T00:00:00",
    end_datetime="2023-12-31T23:59:59",
)
NETCDF = os.path.join(
    DOSSIER, "donnees", "cmems_mod_med_phy-cur_my_4.2km_P1M-m_uo_gibraltar_1993_2023.nc"
)
DERIVE = os.path.join(DOSSIER, "donnees", "gibraltar_transport_mensuel_1993_2023.csv")
WEMO = os.path.join(RACINE, "origine_2026-03", "wemo_index_era5_1993_2023.csv")
SIGNE_WEMO = -1.0
THESE = os.path.join(RACINE, "these")
ENERGIE = os.path.join(THESE, "data", "interim", "eke_mensuelle_twosat.csv")
R_REPRODUIT = 0.230813
SORTIE_BALAYAGE = os.path.join(DOSSIER, "sorties", "gibraltar_inflow_lags.csv")
SORTIE_BILAN = os.path.join(DOSSIER, "sorties", "gibraltar_inflow_summary.csv")
RETARD_MAX = 24
R_TERRE = 6371000.0


def epaisseurs(profondeurs):
    import numpy as np

    z = np.asarray(profondeurs, float)
    bornes = np.empty(z.size + 1)
    bornes[0] = 0.0
    bornes[1:-1] = 0.5 * (z[1:] + z[:-1])
    bornes[-1] = z[-1] + (z[-1] - bornes[-2])
    return np.diff(bornes)


def largeurs(latitudes):
    import numpy as np

    lat = np.asarray(latitudes, float)
    pas = np.gradient(lat) if lat.size > 1 else np.array([1.0])
    return np.abs(pas) * np.pi / 180.0 * R_TERRE


def transports(u, dz, dy):
    import numpy as np

    u = np.nan_to_num(np.asarray(u, float), nan=0.0)
    aire = np.asarray(dz)[:, None] * np.asarray(dy)[None, :]
    entree = (np.where(u > 0, u, 0.0) * aire).sum(axis=(1, 2)) / 1e6
    net = (u * aire).sum(axis=(1, 2)) / 1e6
    return entree, net


def balayage(x, y, retard_max=RETARD_MAX):
    return [
        (k, correlation_au_retard(x, y, k)) for k in range(-retard_max, retard_max + 1)
    ]


def resume(lignes, n_eff):
    s_nom, s_bon = seuils(n_eff, 2 * RETARD_MAX + 1)
    k_max, r_max = max(lignes, key=lambda t: abs(t[1]))
    return {
        "seuil_nominal": s_nom,
        "seuil_bonferroni": s_bon,
        "retard_du_maximum": k_max,
        "r_au_maximum": r_max,
        "r_retard_zero": dict(lignes)[0],
        "retards_nominaux": [k for k, r in lignes if abs(r) > s_nom],
        "retards_bonferroni": [k for k, r in lignes if abs(r) > s_bon],
    }


def telecharger():
    import copernicusmarine

    os.makedirs(os.path.dirname(NETCDF), exist_ok=True)
    reponse = copernicusmarine.subset(
        dataset_id="cmems_mod_med_phy-cur_my_4.2km_P1M-m",
        variables=["uo"],
        minimum_longitude=-5.45,
        maximum_longitude=-5.25,
        minimum_latitude=35.80,
        maximum_latitude=36.20,
        minimum_depth=1.0182366371154785,
        maximum_depth=971.077880859375,
        start_datetime="1993-01-01T00:00:00",
        end_datetime="2023-12-31T23:59:59",
        output_filename=os.path.basename(NETCDF),
        output_directory=os.path.dirname(NETCDF),
        overwrite=True,
    )
    ecrit = getattr(reponse, "file_path", None)
    if not ecrit or (
        os.path.normcase(os.path.abspath(str(ecrit)))
        != os.path.normcase(os.path.abspath(NETCDF))
    ):
        raise SystemExit(f"STOP: the service wrote {ecrit}, not {NETCDF}")
    return 0


def lire_section():
    import numpy as np

    if os.path.isfile(NETCDF):
        import xarray as xr

        ds = xr.open_dataset(NETCDF)
        u = ds["uo"]
        noms = {d.lower(): d for d in u.dims}
        lon, lat, dep, tps = (
            noms.get("longitude"),
            noms.get("latitude"),
            noms.get("depth"),
            noms.get("time"),
        )
        if None in (lon, lat, dep, tps):
            raise SystemExit(f"STOP: unexpected dimensions {u.dims}")
        i = int(np.abs(ds[lon].values - SECTION_LON).argmin())
        col = u.isel({lon: i}).transpose(tps, dep, lat)
        if np.isfinite(col.values[:, -1, :]).any():
            raise SystemExit(
                "STOP: the deepest extracted level is wet; "
                "the water column is truncated"
            )
        dz, dy = epaisseurs(ds[dep].values), largeurs(ds[lat].values)
        entree, net = transports(col.values, dz, dy)
        mois = [str(t)[:7] for t in ds[tps].values]
        lon_section = float(ds[lon].values[i])
        ds.close()
        with io.open(DERIVE, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mois", "entree_sv", "net_sv", "longitude_section"])
            for m, a, b in zip(mois, entree, net):
                w.writerow([m, f"{a:.6f}", f"{b:.6f}", f"{lon_section:.5f}"])
    if not os.path.isfile(DERIVE):
        raise SystemExit(
            f"STOP: neither {NETCDF} nor {DERIVE}; run --telecharger first"
        )
    with io.open(DERIVE, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    return (
        {r["mois"]: float(r["entree_sv"]) for r in lignes},
        {r["mois"]: float(r["net_sv"]) for r in lignes},
        float(lignes[0]["longitude_section"]),
    )


def produire():
    import numpy as np

    entree, net, lon_section = lire_section()
    with io.open(WEMO, encoding="utf-8") as f:
        wemo = {
            r["time"]: SIGNE_WEMO * float(r["wemo_era5"]) for r in csv.DictReader(f)
        }
    if not os.path.isfile(ENERGIE):
        raise SystemExit(f"STOP: thesis series not found: {ENERGIE}")
    with io.open(ENERGIE, encoding="utf-8") as f:
        energie = {
            r["mois"]: float(r["energie_totale_cm2_s2"]) for r in csv.DictReader(f)
        }
    mois = sorted(set(entree) & set(wemo) & set(energie))
    if len(mois) != 372:
        raise SystemExit(f"STOP: {len(mois)} common months instead of 372")
    m12 = [int(m[5:]) for m in mois]
    w = [wemo[m] for m in mois]
    e = desaisonnaliser([energie[m] for m in mois], m12)
    a_in = desaisonnaliser([entree[m] for m in mois], m12)
    a_net = desaisonnaliser([net[m] for m in mois], m12)

    r_rep = correlation_au_retard(w, e, 1)
    if abs(r_rep - R_REPRODUIT) > 1e-6:
        raise SystemExit(
            f"STOP: WeMO/total-flow energy at +1 gives {r_rep:.6f}, "
            f"not {R_REPRODUIT}; nothing is written"
        )

    couples = (
        ("wemo_entree", w, a_in),
        ("wemo_net", w, a_net),
        ("entree_energie_totale", a_in, e),
        ("net_energie_totale", a_net, e),
    )
    bilan, rangs = [], []
    for nom, x, y in couples:
        lignes = balayage(x, y)
        ne = n_effectif(x, y)
        s = resume(lignes, ne)
        bilan.append(
            {
                "couple": nom,
                "n_mois": len(x),
                "n_effectif": f"{ne:.6f}",
                "seuil_nominal": f"{s['seuil_nominal']:.6f}",
                "seuil_bonferroni": f"{s['seuil_bonferroni']:.6f}",
                "r_retard_zero": f"{s['r_retard_zero']:.6f}",
                "retard_du_maximum": s["retard_du_maximum"],
                "r_au_maximum": f"{s['r_au_maximum']:.6f}",
                "retards_nominaux": " ".join(map(str, s["retards_nominaux"]))
                or "aucun",
                "retards_bonferroni": " ".join(map(str, s["retards_bonferroni"]))
                or "aucun",
            }
        )
        rangs += [{"couple": nom, "retard_mois": k, "r": f"{r:.6f}"} for k, r in lignes]
    climat = {
        "entree_moyenne_sv": float(np.mean([entree[m] for m in mois])),
        "net_moyen_sv": float(np.mean([net[m] for m in mois])),
        "longitude_section": lon_section,
    }
    os.makedirs(os.path.dirname(SORTIE_BILAN), exist_ok=True)
    with io.open(SORTIE_BILAN, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(bilan[0]))
        wr.writeheader()
        wr.writerows(bilan)
        f.write("\n")
        f.write("grandeur,valeur\n")
        for k, v in climat.items():
            f.write(f"{k},{v:.6f}\n")
        f.write(f"r_reproduit_wemo_energie_totale_plus1,{r_rep:.6f}\n")
        f.write(f"section_longitude_visee,{SECTION_LON}\n")
        f.write(f"latitude_min,{EMPRISE['minimum_latitude']}\n")
        f.write(f"latitude_max,{EMPRISE['maximum_latitude']}\n")
        f.write(f"profondeur_max_m,{EMPRISE['maximum_depth']}\n")
    with io.open(SORTIE_BALAYAGE, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["couple", "retard_mois", "r"])
        wr.writeheader()
        wr.writerows(rangs)
    return 0


TWOSAT = os.path.join(
    DOSSIER,
    "donnees",
    "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D_multi-vars_"
    "4.88W-9.88E_35.12N-40.88N_1993-01-01-2023-12-31_(1).nc",
)
BANDES = os.path.join(DOSSIER, "donnees", "energie_totale_par_bande_1993_2023.csv")
SORTIE_LONGITUDES = os.path.join(
    DOSSIER, "sorties", "gibraltar_inflow_by_longitude.csv"
)
SORTIE_LONGITUDES_RETARDS = os.path.join(
    DOSSIER, "sorties", "gibraltar_inflow_by_longitude_lags.csv"
)
BORNES_OUEST = list(range(-5, 6))


def bande_de(longitude):
    import math

    return int(math.floor(longitude))


def deriver_bandes(chemin=TWOSAT):
    import numpy as np
    import xarray as xr

    ds = xr.open_dataset(chemin)
    if not {"ugos", "vgos"} <= set(ds.data_vars):
        raise SystemExit(f"STOP: {chemin} does not carry ugos/vgos")
    champ = (0.5 * (ds["ugos"] ** 2 + ds["vgos"] ** 2) * 1e4).resample(time="MS").mean()
    valeurs = champ.transpose("time", "latitude", "longitude").values
    mois = [str(t)[:7] for t in champ["time"].values]
    bandes = np.array([bande_de(x) for x in ds["longitude"].values])
    ds.close()
    with io.open(BANDES, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mois", "bande_ouest", "somme_cm2_s2", "mailles"])
        for t, m in enumerate(mois):
            for b in sorted(set(bandes)):
                bloc = valeurs[t][:, bandes == b]
                ok = np.isfinite(bloc)
                w.writerow([m, b, repr(float(bloc[ok].sum())), int(ok.sum())])


def lire_bandes():
    from collections import defaultdict

    if not os.path.isfile(BANDES):
        raise SystemExit(
            f"STOP: {BANDES} missing; run --deriver-bandes where the TWO-SAT file is"
        )
    d = defaultdict(dict)
    with io.open(BANDES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[r["mois"]][int(r["bande_ouest"])] = (
                float(r["somme_cm2_s2"]),
                int(r["mailles"]),
            )
    return d


def energie_a_l_est_de(bandes, borne, cote="est"):
    garder = (lambda b: b >= borne) if cote == "est" else (lambda b: b < borne)
    out = {}
    for m, par in bandes.items():
        s = sum(v[0] for b, v in par.items() if garder(b))
        n = sum(v[1] for b, v in par.items() if garder(b))
        out[m] = s / n
    return out


def longitudes():
    entree, _, _ = lire_section()
    with io.open(WEMO, encoding="utf-8") as f:
        wemo = {
            r["time"]: SIGNE_WEMO * float(r["wemo_era5"]) for r in csv.DictReader(f)
        }
    with io.open(ENERGIE, encoding="utf-8") as f:
        these = {
            r["mois"]: float(r["energie_totale_cm2_s2"]) for r in csv.DictReader(f)
        }
    bandes = lire_bandes()
    tout = energie_a_l_est_de(bandes, -999)
    ecart = max(abs(tout[m] - these[m]) for m in these)
    if ecart > 1e-6:
        raise SystemExit(
            f"STOP: the union of bands departs from the thesis series by {ecart:.3e}"
        )
    mois = sorted(set(entree) & set(wemo) & set(tout))
    m12 = [int(m[5:]) for m in mois]
    w = [wemo[m] for m in mois]
    a_in = desaisonnaliser([entree[m] for m in mois], m12)
    rangs, retards = [], []
    for cote, borne in [("est", b) for b in BORNES_OUEST] + [
        ("ouest", b) for b in BORNES_OUEST[1:]
    ]:
        e_ser = energie_a_l_est_de(bandes, borne, cote)
        e = desaisonnaliser([e_ser[m] for m in mois], m12)
        mailles = sum(
            v[1]
            for b, v in bandes[mois[0]].items()
            if (b >= borne if cote == "est" else b < borne)
        )
        ligne = {"domaine": cote, "borne_deg_e": borne, "mailles_premier_mois": mailles}
        for nom, x in (("wemo", w), ("entree", a_in)):
            lig = balayage(x, e)
            ne = n_effectif(x, e)
            s = resume(lig, ne)
            r = dict(lig)
            retards += [
                {
                    "domaine": cote,
                    "borne_deg_e": borne,
                    "indice": nom,
                    "retard_mois": k,
                    "r": f"{v:.6f}",
                    "seuil_bonferroni": f"{s['seuil_bonferroni']:.6f}",
                    "seuil_nominal": f"{s['seuil_nominal']:.6f}",
                }
                for k, v in lig
            ]
            ligne.update(
                {
                    f"{nom}_r0": f"{r[0]:.6f}",
                    f"{nom}_r1": f"{r[1]:.6f}",
                    f"{nom}_retard_max": s["retard_du_maximum"],
                    f"{nom}_r_max": f"{s['r_au_maximum']:.6f}",
                    f"{nom}_seuil_bonferroni": f"{s['seuil_bonferroni']:.6f}",
                    f"{nom}_retards_bonferroni": " ".join(
                        map(str, s["retards_bonferroni"])
                    )
                    or "aucun",
                }
            )
        rangs.append(ligne)
    with io.open(SORTIE_LONGITUDES, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rangs[0]))
        wr.writeheader()
        wr.writerows(rangs)
    tab20 = os.path.join(
        DOSSIER, "sorties_these", "these_ch03_tab20_balayage_des_retards.csv"
    )
    with io.open(tab20, encoding="utf-8") as f:
        ref = {
            int(r["retard_mois"]): SIGNE_WEMO * float(r["r"])
            for r in csv.DictReader(f)
            if r["energie"] == "totale_quotidienne"
            and r["serie"] == "reconstruction d'origine (signe inverse)"
        }
    ici = {
        r["retard_mois"]: float(r["r"])
        for r in retards
        if r["domaine"] == "est" and r["borne_deg_e"] == -5 and r["indice"] == "wemo"
    }
    ecart = (
        max(abs(ref[k] - ici[k]) for k in ici) if len(ref) == len(ici) == 49 else None
    )
    if ecart is None or ecart > 2e-6:
        raise SystemExit(
            f"STOP: the whole-domain lag sweep does not reproduce ch03_tab20 ({ecart})"
        )
    with io.open(SORTIE_LONGITUDES_RETARDS, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(retards[0]))
        wr.writeheader()
        wr.writerows(retards)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--telecharger" in a:
        sys.exit(telecharger())
    if "--produire" in a:
        sys.exit(produire())
    if "--deriver-bandes" in a:
        sys.exit(deriver_bandes())
    if "--longitudes" in a:
        sys.exit(longitudes())
    print(__doc__)
