
import argparse
import yaml
import os
import shutil
import json
import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch


from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


from fault_management_uds.config import Config, get_additional_configurations
from fault_management_uds.utilities import folder_cleanup, seed_everything
from fault_management_uds.data.dataset import load_data, get_datasets, handle_splits, identify_valid_indices
from fault_management_uds.modelling.models import get_model, load_model_checkpoint
from fault_management_uds.modelling.train import train_model
from fault_management_uds.plots import visualize_logs
from fault_management_uds.modelling.evaluate import evaluate_model_on_dataset


from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR


# Set default global settings for tqdm
tqdm_kwargs = {'mininterval': 60.0}  # Updates every _ seconds
# Replace the tqdm function globally with pre-configured options
import builtins
builtins.tqdm = lambda *args, **kwargs: tqdm(*args, **{**tqdm_kwargs, **kwargs})


def count_parameters(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: The total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--save_folder', type=str, default=None, help='Save folder path')
    parser.add_argument('--predict_steps_ahead', type=int, default=10, help='Number of steps ahead to predict')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()


def main():

    ### Set up
    # Parse arguments
    args = parse_args()


    # set save folder to path object
    save_folder = MODELS_DIR / args.save_folder

    split = 0 # first split
    relative_path = f'{split+1}_split/'
    
    run_folder = save_folder / relative_path
    eval_folder = run_folder / 'evaluation'

    # Get relevant indo
    config = yaml.load(open(save_folder / 'config.yaml', 'r'), Loader=yaml.Loader)
    config['num_workers'] = args.num_workers
    config['predict_steps_ahead'] = args.predict_steps_ahead
    # load pkl _with cpu
    split_info = torch.load(save_folder / 'split_info.pkl', map_location='cpu')
    run_info = split_info[split]



    # Load configuration
    seed_everything(config['training_args']['seed'])

    ### Load data
    data = load_data([None, None], config['dataset_args']['data_file_path'], config['dataset_args'], 
                        data_type='complete',
                        data_group=config['dataset_args'].get('data_group', "clean"),
                        )

    ### Iterate over the different splits
    train_index, val_index, test_index = run_info['dataset_config']['train_index'], run_info['dataset_config']['val_index'], run_info['dataset_config']['test_index']

    ### Prepare data
    train_dataset, val_dataset, test_dataset, dataset_config = get_datasets(data, train_index, val_index, test_index, config.config['dataset_args'])

    # Create loader
    sampler = WeightedRandomSampler(train_dataset.valid_priority_weight, len(train_dataset), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config['training_args']['batch_size'], sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['training_args']['batch_size'], shuffle=False, num_workers=args.num_workers)


    ### Train model
    additional_configurations = get_additional_configurations(train_dataset)

    ### Evaluate model
    # load the model: ['best_model_path', 'last_model_path']
    model_to_load = config['training_args']['model_to_load']
    model = load_model_checkpoint(run_folder, run_info, model_to_load, config, additional_configurations)

    
    # evaluate the model
    # train set
    train_dataset.valid_indices = identify_valid_indices(train_dataset.not_nan_mask, train_dataset.sequence_length, config['predict_steps_ahead'])
    train_dataset.valid_timestamps = pd.to_datetime(train_dataset.timestamps[train_dataset.valid_indices.cpu().numpy()])
    evaluate_model_on_dataset(eval_folder / 'train', model, train_dataset, dataset_config['scalers'], config, data_type='train')
    # validation set, update the valid indices and timestamps wrt forecast horizon
    val_dataset.valid_indices = identify_valid_indices(val_dataset.not_nan_mask, val_dataset.sequence_length, config['predict_steps_ahead'])
    val_dataset.valid_timestamps = pd.to_datetime(val_dataset.timestamps[val_dataset.valid_indices.cpu().numpy()])
    evaluate_model_on_dataset(eval_folder / 'val', model, val_dataset, dataset_config['scalers'], config, data_type='val')
    # test set
    test_dataset.valid_indices = identify_valid_indices(test_dataset.not_nan_mask, test_dataset.sequence_length, config['predict_steps_ahead'])
    test_dataset.valid_timestamps = pd.to_datetime(test_dataset.timestamps[test_dataset.valid_indices.cpu().numpy()])
    evaluate_model_on_dataset(eval_folder / 'test', model, test_dataset, dataset_config['scalers'], config, data_type='test')

    print('')


    print(f"\n###   Run {config['experiment_name']} completed   ###\n")


    print("\n")






if __name__ == '__main__':
    main()











