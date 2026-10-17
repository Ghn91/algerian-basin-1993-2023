"""Download the two altimetry products used in the revised analysis:
CMEMS DUACS DT-2024 ALL-SAT (0.125 deg) and C3S DUACS DT-2024 TWO-SAT (0.25 deg),
daily, 1993-2023, over the study domain. Requires Copernicus credentials
(copernicusmarine login; ~/.cdsapirc).

Usage: python download_altimetry.py [--allsat | --twosat | --catalogue]
"""

import os
import sys
from pathlib import Path

DONNEES = Path(os.environ.get("ABASIN_DATA", Path(__file__).parent / "donnees"))

EMPRISE = dict(
    minimum_longitude=-4.94,
    maximum_longitude=9.94,
    minimum_latitude=35.06,
    maximum_latitude=40.94,
)
DEBUT, FIN = "1993-01-01", "2023-12-31"


def catalogue():
    import copernicusmarine

    trouve = {}
    for motif in ("sea level", "sealevel", "duacs", "sea surface height"):
        try:
            cat = copernicusmarine.describe(contains=[motif])
        except TypeError:
            cat = copernicusmarine.describe()
        for p in getattr(cat, "products", cat) or []:
            pid = getattr(p, "product_id", "")
            for d in getattr(p, "datasets", None) or []:
                i = getattr(d, "dataset_id", "")
                if i.startswith("cmems_obs-sl") or "SEALEVEL" in pid.upper():
                    trouve[i] = pid
    if not trouve:
        return


def allsat():
    import copernicusmarine

    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla", "ugosa", "vgosa", "adt", "ugos", "vgos"],
        start_datetime=f"{DEBUT}T00:00:00",
        end_datetime=f"{FIN}T00:00:00",
        output_directory=str(DONNEES),
        **EMPRISE,
    )


TWOSAT_ID = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"


def twosat():
    import copernicusmarine

    copernicusmarine.subset(
        dataset_id=TWOSAT_ID,
        variables=["sla", "ugosa", "vgosa", "adt", "ugos", "vgos"],
        start_datetime=f"{DEBUT}T00:00:00",
        end_datetime=f"{FIN}T00:00:00",
        output_directory=str(DONNEES),
        **EMPRISE,
    )


def main():
    DONNEES.mkdir(parents=True, exist_ok=True)
    if "--catalogue" in sys.argv:
        catalogue()
        return
    faire_tout = not ({"--allsat", "--twosat"} & set(sys.argv))
    if faire_tout or "--allsat" in sys.argv:
        allsat()
    if faire_tout or "--twosat" in sys.argv:
        twosat()


if __name__ == "__main__":
    main()
