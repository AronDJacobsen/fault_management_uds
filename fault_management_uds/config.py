from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REFERENCE_DIR = PROJ_ROOT / "references"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


error_indicators = ['man_remove', 'stamp', 'outbound', 'outlier', 'frozen'] # TODO: frozen high??


bools_2_meta = {
    'ffill': {
        'color': 'darkkhaki',   
        'alias': 'Data-saving',
    },
    'man_remove': {
        'color': 'mediumslateblue',
        'alias': 'Removed',
    },
    'stamp': {
        'color': 'slategray',
        'alias': 'Stamped',
    },
    'outbound': {
        'color': 'lightcoral',
        'alias': 'Outbound',
    },
    'outlier': {
        'color': 'tomato',
        'alias': 'Outlier',
    },
    'frozen': {
        'color': 'lightskyblue',
        'alias': 'Frozen',
    },
}

indicator_2_meta = {
    0: {
        'color': 'antiquewhite',
        'alias': 'No Data',
    },
    1: {
        'color': 'forestgreen',
        'alias': 'Data',
    },
    -1: {
        'color': 'lightgreen',
        'alias': 'Obvious min.',
    },
    2: {
        'color': 'firebrick',
        'alias': 'Erroneous',
    },
}


rain_gauges = [
    '5425',
    '5427'
]


structure_2_sensor = {
    # Brændekilde
    'G80F11B': [
        'G80F11B_Level1', # inlet, can overlow over a wall (inlet/enter), and river?
        'G80F11B_Level2' # inlet, can overlow over a wall (inlet/enter), and river?
        ],
    'G80F66Y': [
        'G80F66Y_Level1', # basin, can overflow over a wall and store water
        'G80F66Y_Level2' # basin, can overflow over a wall and store water
        ],
    'G80F13P': [
        'G80F13P_LevelPS', # meters high, this is where a pump is located
        # a pumping station has two pumps?
        'G80F13Pp1_power',
        'G80F13Pp2_power',
        ],


    'G73F010': [
        'G73F010', # pipe? level
        ],
    'G72F040': [
        'G72F040', # pipe? level
        ],

    # Bellinge
    'G71F05R': [
        'G71F05R_LevelInlet', # inlet, can overflow into the basin
        'G71F05R_LevelBasin',  # basin, can overflow over a wall and into the storage pipe
        'G71F05R_position', # closed at start of rain, then opens to flush the storage pipe
        ],

    'G71F04R': [
        'G71F04R_Level1', # inlet, can overflow into the storage pipe
        'G71F04R_Level2', # inlet, can overflow into the storage pipe
        ],

    # Dyrup
    'G71F06R': [
        'G71F06R_LevelInlet', # inlet, can overflow into the storage pipe
        ],

    # storage pipe from Bellinge to Dyrup
    'G71F68Y': [
        'G71F68Y_LevelPS', # meters high, this is where a pump is located
        'G71F68Yp1', # the flow in the pump up to Dyrup (G71F06R)
        'G71F68Yp1_power',
        'G71F68Yp2_power',
        ]
}


natural_structure_order = list(structure_2_sensor.keys())

natural_sensor_order = []
for structure in natural_structure_order:
    natural_sensor_order += structure_2_sensor[structure]


single_series_order = rain_gauges + natural_sensor_order



sensor_2_alias = {

    # Brændekilde
    'G80F11B_Level1': 'Basin Level (1)',
    'G80F11B_Level2': 'Basin Level (2)',

    'G80F66Y_Level1': 'Inlet Level (1)',
    'G80F66Y_Level2': 'Inlet Level (2)',

    'G80F13P_LevelPS': 'Pump Sump Level',
    'G80F13Pp1_power': 'Pump 1 Power',
    'G80F13Pp2_power': 'Pump 2 Power',

    # Bellinge, pipe
    'G73F010': 'Pipe Level',

    'G72F040': 'Pipe Level',

    # Bellinge
    'G71F05R_LevelInlet': 'Inlet Level',
    'G71F05R_LevelBasin': 'Basin Level',
    'G71F05R_position': 'Throttle Position',


    'G71F04R_Level1': 'Pipe Level (1)',
    'G71F04R_Level2': 'Pipe Level (2)',

    # Dyrup
    'G71F06R_LevelInlet': 'Inlet Level',

    # storage pipe from Bellinge to Dyrup
    'G71F68Y_LevelPS': 'Pump Sump Level',
    'G71F68Yp1': 'Pump 1 Flow',
    'G71F68Yp1_power': 'Pump 1 Power',
    'G71F68Yp2_power': 'Pump 2 Power',
}



