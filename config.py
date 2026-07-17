"""Central configuration: one place to point at your data.

Edit DATA_DIR below, or set the environment variable ABASIN_DATA.
All scripts import DATA_DIR / BASE / FIG_DIR / EDDY_DIR / TEMP_DIR from here.
"""
import os

DATA_DIR = os.environ.get("ABASIN_DATA", r"C:\Users\kikk\CLAUDE ARTICLE DATA")
BASE     = DATA_DIR                                   # alias used by some scripts
FIG_DIR  = os.path.join(DATA_DIR, "FIGURES", "PUBLICATION")
EDDY_DIR = os.path.join(DATA_DIR, "EDDIES")
TEMP_DIR = os.path.join(EDDY_DIR, "TEMP")

# create the output sub-folders only if the data root actually exists,
# so importing config on a machine without the data does no harm
if os.path.isdir(DATA_DIR):
    for _d in (FIG_DIR, EDDY_DIR, TEMP_DIR):
        os.makedirs(_d, exist_ok=True)
