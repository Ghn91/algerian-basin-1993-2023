import os

# path to the folder holding the downloaded NetCDF files
DATA_DIR = os.environ.get("ABASIN_DATA", r"C:\Users\kikk\CLAUDE ARTICLE DATA")
BASE     = DATA_DIR
FIG_DIR  = os.path.join(DATA_DIR, "FIGURES")
EDDY_DIR = os.path.join(DATA_DIR, "EDDIES")
TEMP_DIR = os.path.join(EDDY_DIR, "TEMP")

for d in (FIG_DIR, EDDY_DIR, TEMP_DIR):
    os.makedirs(d, exist_ok=True)
