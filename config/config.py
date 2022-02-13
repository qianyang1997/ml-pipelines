import logging
from pathlib import Path
import pandas as pd
import yaml

# set paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
_CONFIG_PATH = ROOT_DIR / "config/params.yml"
_CREDS_PATH = ROOT_DIR / "config/creds.yml"

# configurate pandas output
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('chained_assignment', None)

# load yaml files
with open(_CONFIG_PATH, "r") as f:
    _CONFIG = yaml.safe_load(f)
with open(_CREDS_PATH, "r") as f:
    _CREDS = yaml.safe_load(f)
PATH = _CONFIG.fetch('filepath')
PARAMS = _CONFIG.fetch('model')

# set filepaths
for key in PATH:
    directory = PATH[key].fetch('directory')
    files = PATH[key].fetch('files')
    if (not directory) and files:
        raise FileNotFoundError("Directory not specified in yaml file.")
    for file in files:
        PATH[key]['files'][file] = ROOT_DIR / directory / file

# logging