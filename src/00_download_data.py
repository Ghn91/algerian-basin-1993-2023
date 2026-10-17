"""
Download all CMEMS / C3S datasets via the copernicusmarine toolbox.

Part of the reproducibility archive for:
Nedjar, Nacef, Houma & Bachari (in prep.), "Thirty years of surface
circulation variability in the Algerian Basin (1993-2023)",
Regional Studies in Marine Science.

Data root and derived paths come from config.py (edit DATA_DIR there,
or set the ABASIN_DATA environment variable).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, BASE  # noqa: F401

import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_phy-temp_my_4.2km_P1D-m",
  variables=["thetao"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=1.0182366371154785,
  maximum_depth=1.0182366371154785,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_phy-sal_my_4.2km_P1D-m",
  variables=["so"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=1.0182366371154785,
  maximum_depth=1.0182366371154785,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_phy-ssh_my_4.2km_P1D-m",
  variables=["zos"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_phy-mld_my_4.2km_P1D-m",
  variables=["mlotst"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
  variables=["thetao", "uo", "vo", "so", "zos", "mlotst"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=0.49402499198913574,
  maximum_depth=0.49402499198913574,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_bgc-bio_my_4.2km_P1D-m",
  variables=["o2"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1999-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=1.0182366371154785,
  maximum_depth=1.0182366371154785,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_bgc-nut_my_4.2km_P1D-m",
  variables=["no3", "po4"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1999-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=1.0182366371154785,
  maximum_depth=1.0182366371154785,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_mod_med_bgc-plankton_my_4.2km_P1D-m",
  variables=["chl", "phyc"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1999-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
  minimum_depth=1.0182366371154785,
  maximum_depth=1.0182366371154785,
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021",
  variables=["analysed_sst", "analysis_error"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="1993-01-01T00:00:00",
  end_datetime="2023-12-31T00:00:00",
)

# ----------------------------------------------------------------------
import copernicusmarine

copernicusmarine.subset(
  dataset_id="cmems_obs-ins_med_phybgcwav_mynrt_na_irr",
  dataset_part="monthly",
  variables=["TEMP", "PSAL", "DOXY"],
  minimum_longitude=-5,
  maximum_longitude=10,
  minimum_latitude=35,
  maximum_latitude=41,
  start_datetime="2020-01-01T12:00:00",
  end_datetime="2023-12-31T12:00:00",
  minimum_depth=-20,
  maximum_depth=20,
)
