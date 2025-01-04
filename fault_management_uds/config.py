from pathlib import Path
import yaml
import itertools
from datetime import datetime
import os

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

condition_to_meta = {
    'Overall': {
        'color': 'gray',
    },
    'Rain': {
        'color': 'lightskyblue',
    },
    'Extreme': {
        'color': 'violet',
    },
    'Dry': {
        'color': 'bisque',
    },
}


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


rain_gauges = ['5425', '5427']

rain_gauge_color = {
    '5425': 'darkblue',
    '5427': 'purple',
}


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




class Config:
    """
    Class to store configuration parameters
    """
    def __init__(self, config_path, fast_run=False, save_folder=None, num_workers=0):
        # Define experiment folder; given 'transformer/2_data_features/test.yaml' we want 'transformer/2_data_features'
        self.relative_path = '/'.join(config_path.split('/')[:-1])
        if save_folder is not None:
            self.experiment_folder = save_folder / self.relative_path # config_path.split('/')[0].split('.')[0]
        else:
            self.experiment_folder = MODELS_DIR / self.relative_path # config_path.split('/')[0].split('.')[0]
        os.makedirs(self.experiment_folder, exist_ok=True)
        self.num_workers = num_workers
        
        # Load configuration
        self.config_folder = PROJ_ROOT / 'experiments'
        self.config_path = self.config_folder / config_path
        self.config = self.load_config()

        # Define additional parameters
        self.define_additional_parameters()

        # Define the hyperparameter search grid
        self.hparam_grid = self.get_hparam_grid()

        # Update with hyperparameters
        if fast_run:
            # update parameters
            self.config['training_args']['max_epochs'] = -1
            self.config['training_args']['max_steps'] = 100
            self.config['training_args']['log_every_n_steps'] = 1
            self.config['training_args']['val_check_interval'] = 5

    def define_additional_parameters(self):
        self.config['dataset_args']['sequence_length'] = self.config['model_args']['sequence_length']
        self.config['dataset_args']['data_file_path'] = PROCESSED_DATA_DIR / 'Bellinge.h5'
        self.config['dataset_args']['variable_list'] = self.config['dataset_args']['engineered_vars'] + self.config['dataset_args']['exogenous_vars'] + self.config['dataset_args']['endogenous_vars']
        self.config['dataset_args']['data_variables'] = self.config['dataset_args']['exogenous_vars'] + self.config['dataset_args']['endogenous_vars']
        self.config['model_args']['input_size'] = len(self.config['dataset_args']['variable_list']) - len(self.config['dataset_args']['hold_out_vars'])
        self.config['model_args']['output_size'] = len(self.config['dataset_args']['endogenous_vars'])


    def load_config(self, ):
        """
        Load configuration from a YAML file
        """

        # load the default configuration
        with open(self.config_folder / 'default.yaml', 'r') as f:
            config = yaml.safe_load(f)

        # load base configuration if it exists in the config path folder, go one back
        config_path_folder = self.config_path.parents[0]
        if (config_path_folder / 'base.yaml').exists():
            with open(config_path_folder / 'base.yaml', 'r') as f:
                base_config = yaml.safe_load(f)
            config = self.update_config(config, base_config)

        # load the experiment configuration
        with open(self.config_path, 'r') as f:
            experiment_config = yaml.safe_load(f)
        config = self.update_config(config, experiment_config)

        return config


    def update_config(self, config, new_config):
        """
        Update the configuration with new configuration
        """
        for key, value in new_config.items():
            if isinstance(value, dict):
                config[key] = self.update_config(config.get(key, {}), value)
            else:
                config[key] = value
        return config

    def get_hparam_grid(self):
        # If no hyperparameters are defined, return an empty list
        if self.config['hparam_key_paths'] is None:
            # set the hparam to model name
            #self.config['hparam_key_paths'] = [f"model_args/{self.config['model_args']['model_name']}"]
            raise ValueError("No hyperparameters defined; must define hparam_key_paths in the configuration")
        
        # Create a dictionary with the hyperparameters
        param_grid = {}
        for hparam in self.config['hparam_key_paths']:
            # get the list of values
            param_grid[hparam] = self.get_value_from_key_list(hparam.split('/'))
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        hparam_grid = [dict(zip(keys, combo)) for combo in combinations]
        # create names
        return hparam_grid

    def get_value_from_key_list(self, key_list):
        value = self.config
        for key in key_list:
            value = value[key]
        # ensure it is a list
        if not isinstance(value, list):
            print(f"Warning: {key_list} is not a list; must be for hparam search")
            value = [value]
        return value


    def update_with_hparams(self, hparams):
        """
        Update the configuration with hyperparameters
        """
        experiment_name = ''
        for keys_str, value in hparams.items():
            keys = keys_str.split('/')
            new_config = self.get_dict_from_key_list_and_value(keys, value)
            self.config = self.update_config(self.config, new_config)
            experiment_name += f"{keys[-1]}={value}_"
        # Define additional parameters
        self.define_additional_parameters()
                
        # add the time, e.g. "210911", "yymmdd"
        self.config['experiment_name'] = experiment_name[:-1] + f"_{datetime.now().strftime('%y%m%d_%H%M')}"
        self.config['save_folder'] = self.experiment_folder / self.config['experiment_name']
        os.makedirs(self.config['save_folder'], exist_ok=True)

    def get_dict_from_key_list_and_value(self, key_list, value):
        # base case
        if len(key_list) == 1:
            return {key_list[0]: value}
        else:
            return {key_list[0]: self.get_dict_from_key_list_and_value(key_list[1:], value)}

    def save_config(self, save_folder):
        # Save as yaml
        with open(save_folder / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)



def get_additional_configurations(dataset):
    additional_configurations = {}
    # mean
    additional_configurations['target_mean'] = dataset.data[dataset.valid_indices, dataset.endogenous_idx].mean()
    # endogenous_idx
    additional_configurations['endogenous_idx'] = dataset.endogenous_idx
    # the number of data points
    additional_configurations['n_obs'] = len(dataset)
    return additional_configurations
  



