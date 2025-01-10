
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


def handle_anomalous_iteration(config, fine_tune_path, dataset, data_type):
    # load the required dataset indices from the previous model

    anomalous_iteration = config['dataset_args']['anomalous_iteration']
    # load the required dataset indices
    relative_path = '1_split/anomalous'
    model_path = MODELS_DIR / fine_tune_path
    save_path = model_path / relative_path / data_type
    # assert that the folder exists
    assert save_path.exists(), f"Anomalous folder {save_path} does not exist."
    # load the anomaly_prediction_results.pkl and relevant indices for the datasets
    selected_valid_indices = {}
    with open(save_path / 'anomaly_prediction_results.pkl', 'rb') as f:
        anomaly_prediction_results = pickle.load(f)
    # extract relevant indices for selected anomalous iteration
    selected_valid_indices = anomaly_prediction_results[str(anomalous_iteration)]
    # update the valid indices for the datasets
    dataset.update_valid_indices(selected_valid_indices)
    return dataset



def load_model_to_fine_tune(fine_tune_path, additional_configurations):
    split = 0
    fine_tune_path = MODELS_DIR / fine_tune_path
    print(f"Loding fine-tuned model from {fine_tune_path}")
    assert fine_tune_path.exists(), f"Model folder {fine_tune_path} does not exist."
    # get configs
    config = yaml.load(open(fine_tune_path / 'config.yaml', 'r'), Loader=yaml.Loader)
    split_info = torch.load(fine_tune_path / 'split_info.pkl')#, map_location='cpu')
    run_info = split_info[split]   
    # load the model
    model_to_load = config['training_args']['model_to_load']
    model = load_model_checkpoint(fine_tune_path / f"{split+1}_split", run_info, model_to_load, config, additional_configurations)
    # since fine-tuning, set the model to train mode
    model.train()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--config', type=str, default='default.yaml', help='config file')
    parser.add_argument('--fine_tune_path', type=str, default=None, help='Fine-tune path')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--save_folder', type=str, default=None, help='Save folder')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()


def main():

    ### Set up
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config(config_path=args.config, save_folder=args.save_folder, fine_tune_path=args.fine_tune_path, fast_run=args.fast_run, num_workers=args.num_workers)
    
    seed_everything(config.config['training_args']['seed'])

    # # Clean up folders
    # folder_cleanup(MODELS_DIR)
    # folder_cleanup(config.experiment_folder)
    
    # Create experiment folder if it does not exist
    if not os.path.exists(config.experiment_folder):
        os.makedirs(config.experiment_folder)


    ### Iterate over the different experiments (based on defined hyperparameters)
    print('')
    for i, hparams in enumerate(config.hparam_grid):
        print("#" * 50)
        print(f"EXPERIMENT {i+1}/{len(config.hparam_grid)}")
        # Define additional parameters
        config.update_with_hparams(hparams)
        print(f"Save folder: {config.config['save_folder']}")
        print(f"\nHyperparameters: {hparams}")
        print("#" * 50)
        print('')


        ### Load data
        data = load_data([None, None], config.config['dataset_args']['data_file_path'], config.config['dataset_args'], 
                         data_type='complete',
                         data_group=config.config['dataset_args']['data_group']
                         )
        data = data if not args.fast_run else data.iloc[:3000]
        n_obs = len(data)

        # Handle splits
        splits = handle_splits(n_obs, config.config['dataset_args'])
        config.config['split_folders'] = [config.config['save_folder'] / f"{i+1}_split" for i in range(config.config['dataset_args']['n_splits'])]

        # Save the configuration as yaml
        config.save_config(config.config['save_folder'])

        ### Iterate over the different splits
        split_info = []
        for i, (train_index, val_index, test_index) in enumerate(tqdm(splits, desc='Cross-validation', total=len(splits))):
            #print(f"\nSplit {i+1}/{len(splits)}")
            
            # Paths
            current_save_folder = config.config['split_folders'][i]
            # create the folder
            current_save_folder.mkdir(exist_ok=True)
            start_time = time.time()   


            ### Prepare data
            train_dataset, val_dataset, test_dataset, dataset_config = get_datasets(data, train_index, val_index, test_index, config.config['dataset_args'])
            # handle anomalous iteration if specified
            if config.config['dataset_args']['anomalous_iteration'] != None:
                train_dataset = handle_anomalous_iteration(config.config, args.fine_tune_path, train_dataset, 'train')
                val_dataset = handle_anomalous_iteration(config.config, args.fine_tune_path, val_dataset, 'val')
                test_dataset = handle_anomalous_iteration(config.config, args.fine_tune_path, test_dataset, 'test')
            # Create loader
            sampler = WeightedRandomSampler(train_dataset.valid_priority_weight, len(train_dataset), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=config.config['training_args']['batch_size'], sampler=sampler, num_workers=config.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=config.config['training_args']['batch_size'], shuffle=False, num_workers=config.num_workers)


            ### Train model
            additional_configurations = get_additional_configurations(train_dataset)
            
            # load fine-tuned model if specified
            if args.fine_tune_path is None:
                model = get_model(config.config['model_args'], config.config['training_args'], additional_configurations=additional_configurations)
            else:
                model = load_model_to_fine_tune(args.fine_tune_path, additional_configurations)

            # Define callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=current_save_folder, filename="{epoch:02d}-{val_loss:.7f}", save_last=True,
                monitor="val_loss", save_top_k=1, mode="min",
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=config.config['training_args']['early_stopping_patience'], mode="min", check_on_train_epoch_end=True, verbose=False)
            #custom_progress_bar = CustomProgressBar()
            custom_progress_bar = TQDMProgressBar(refresh_rate=5000)
            callbacks = [checkpoint_callback, early_stopping, custom_progress_bar]
            # logger
            logger = TensorBoardLogger(current_save_folder, sub_dir='', name='', version=0, default_hp_metric=False)
            logger.log_hyperparams({"placeholder_param": 1})

            # train model
            model, callbacks, logger = train_model(model, train_loader, val_loader, callbacks, logger, config.config['training_args'], current_save_folder)
            # save the run info
            run_info = {
                'save_folder': str(current_save_folder),
                'best_model_path': os.path.relpath(callbacks[0].best_model_path, current_save_folder),
                'last_model_path': os.path.relpath(callbacks[0].last_model_path, current_save_folder),
                # 'top_k_best_model_paths': {
                #     k: os.path.relpath(str(v), current_save_folder) for k, v in callbacks[0].best_k_models.items()
                # },
                'dataset_config': dataset_config,
                'training_time': (time.time() - start_time) / 60,  # in minutes,
            }
            split_info.append(run_info)


            ### Evaluate model
            # load the model: ['best_model_path', 'last_model_path']
            model_to_load = config.config['training_args']['model_to_load']
            model = load_model_checkpoint(current_save_folder, run_info, model_to_load, config.config, additional_configurations)

            # Visualize the training logs
            visualize_logs(current_save_folder / 'version_0')
            
            # evaluate the model
            eval_folder = current_save_folder / 'evaluation'
            eval_folder.mkdir(exist_ok=True)
            # train set
            train_dataset.valid_indices = identify_valid_indices(train_dataset.not_nan_mask, train_dataset.sequence_length, config.config['predict_steps_ahead'])
            train_dataset.valid_timestamps = pd.to_datetime(train_dataset.timestamps[train_dataset.valid_indices.cpu().numpy()])
            evaluate_model_on_dataset(eval_folder / 'train', model, train_dataset, dataset_config['scalers'], config, data_type='train')
            # validation set, update the valid indices and timestamps wrt forecast horizon
            val_dataset.valid_indices = identify_valid_indices(val_dataset.not_nan_mask, val_dataset.sequence_length, config.config['predict_steps_ahead'])
            val_dataset.valid_timestamps = pd.to_datetime(val_dataset.timestamps[val_dataset.valid_indices.cpu().numpy()])
            evaluate_model_on_dataset(eval_folder / 'val', model, val_dataset, dataset_config['scalers'], config, data_type='val')
            # test set
            test_dataset.valid_indices = identify_valid_indices(test_dataset.not_nan_mask, test_dataset.sequence_length, config.config['predict_steps_ahead'])
            test_dataset.valid_timestamps = pd.to_datetime(test_dataset.timestamps[test_dataset.valid_indices.cpu().numpy()])
            evaluate_model_on_dataset(eval_folder / 'test', model, test_dataset, dataset_config['scalers'], config, data_type='test')

            print('')


        # Save the split_info using torch.save (CPU-safe by default)
        torch.save(split_info, config.config['save_folder'] / 'split_info.pkl')

        print(f"\n###   Run {config.config['experiment_name']} completed   ###\n")


        print("\n")






if __name__ == '__main__':
    main()
