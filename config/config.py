import logging
import warnings
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
PATH = _CONFIG['filepath']
MODEL_PARAMS = _CONFIG['model']
VAL_PARAMS = _CONFIG['validation']

# set filepaths
for key in PATH:
    directory = PATH[key].get('directory', [])
    files = PATH[key].get('files', [])
    if (not directory) and files:
        raise FileNotFoundError("Directory not specified in yaml file.")
    if files:
        for file in files:
            PATH[key]['files'][file] = ROOT_DIR / directory / files[file]

# logging
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)

file = logging.FileHandler(filename=ROOT_DIR / 'logs/ml_pipeline.log')
file.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[console, file]
                    )

# warnings
warnings.filterwarnings(action='ignore', category=UserWarning)




