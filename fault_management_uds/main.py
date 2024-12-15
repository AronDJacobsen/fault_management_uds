
import argparse
import os
import shutil
import json
import time
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd


from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from fault_management_uds.config import Config
from fault_management_uds.utilities import folder_cleanup, seed_everything
from fault_management_uds.data.dataset import load_data, get_datasets, handle_splits, identify_valid_indices
from fault_management_uds.modelling.models import get_model
from fault_management_uds.modelling.train import train_model
from fault_management_uds.plots import visualize_logs
from fault_management_uds.modelling.evaluate import evaluate_model_on_dataset
from fault_management_uds.modelling.models import load_model_checkpoint


from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR


def get_additional_configurations(dataset):
    additional_configurations = {}
    # mean
    additional_configurations['target_mean'] = dataset.data[dataset.valid_indices, dataset.endogenous_idx].mean()
    # endogenous_idx
    additional_configurations['endogenous_idx'] = dataset.endogenous_idx
    return additional_configurations



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--config', type=str, default='default.yaml', help='config file')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--save_folder', type=str, default=None, help='Save folder')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()


def main():

    ### Set up
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = Config(args.config, fast_run=args.fast_run, save_folder=args.save_folder, num_workers=args.num_workers)
    seed_everything(config.config['training_args']['seed'])

    # Clean up folders
    folder_cleanup(MODELS_DIR)
    folder_cleanup(config.experiment_folder)
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
        data = load_data([None, None], config.config['dataset_args']['data_file_path'], config.config['dataset_args'], data_type='complete')
        data = data if not args.fast_run else data.sample(3000)
        n_obs = len(data)

        # Handle splits
        splits = handle_splits(n_obs, config.config['dataset_args'])
        config.config['split_folders'] = [config.config['save_folder'] / f"{i+1}_split" for i in range(config.config['dataset_args']['n_splits'])]
        for folder in config.config['split_folders']:
            os.makedirs(folder, exist_ok=True)

        # Save the configuration as yaml
        config.save_config(config.config['save_folder'])


        ### Iterate over the different splits
        split_info = []
        for i, (train_index, val_index, test_index) in enumerate(tqdm(splits, desc='Cross-validation', total=len(splits))):
            #print(f"\nSplit {i+1}/{len(splits)}")
            
            # Paths
            current_save_folder = config.config['split_folders'][i]
            start_time = time.time()   


            ### Prepare data
            train_dataset, val_dataset, test_dataset, dataset_config = get_datasets(data, train_index, val_index, test_index, config.config['dataset_args'])
            # create loader
            sampler = WeightedRandomSampler(train_dataset.priority_weight, len(train_dataset), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=config.config['training_args']['batch_size'], sampler=sampler, num_workers=config.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=config.config['training_args']['batch_size'], shuffle=False, num_workers=config.num_workers)


            ### Train model
            additional_configurations = get_additional_configurations(train_dataset)
            model = get_model(config.config['model_args'], config.config['training_args'], additional_configurations=additional_configurations)
            # configure the model based on the dataset
            model.additional_configurations(train_dataset)

            # Define callbacks
            checkpoint_callback = ModelCheckpoint(
                dirpath=current_save_folder, filename="{epoch:02d}-{val_loss:.6f}", save_last=True,
                monitor="val_loss", save_top_k=1, mode="min",
            )
            early_stopping = EarlyStopping(monitor="val_loss", patience=config.config['training_args']['early_stopping_patience'], mode="min", verbose=False)
            callbacks = [checkpoint_callback, early_stopping]
            # logger
            logger = TensorBoardLogger(current_save_folder, sub_dir='', name='', version=0, default_hp_metric=False)

            # train model
            model, callbacks, logger = train_model(model, train_loader, val_loader, callbacks, logger, config.config['training_args'], current_save_folder)
            training_time = (time.time() - start_time) / 60  # in minutes
            # save the run info
            run_info = {
                'save_folder': str(current_save_folder),
                'best_model_path': callbacks[0].best_model_path,
                'last_model_path': callbacks[0].last_model_path,
                'top_k_best_model_paths': callbacks[0].best_k_models,
                'dataset_config': dataset_config,
                'training_time': (time.time() - start_time) / 60,  # in minutes,
            }
            split_info.append(run_info)


            ### Evaluate model
            # load the model
            model_to_load = 'best_model_path' # ['best_model_path', 'last_model_path']
            model = load_model_checkpoint(current_save_folder, run_info, model_to_load, config.config)

            # Visualize the training logs
            visualize_logs(current_save_folder / 'version_0')
            
            # evaluate the model
            eval_folder = current_save_folder / 'evaluation'
            eval_folder.mkdir(exist_ok=True)
            # validation set
            val_dataset.valid_indices = identify_valid_indices(val_dataset.not_nan_mask, val_dataset.sequence_length, config.config['predict_steps_ahead'])
            val_dataset.valid_timestamps = pd.to_datetime(val_dataset.timestamps[val_dataset.valid_indices.cpu().numpy()])
            evaluate_model_on_dataset(eval_folder / 'val', model, val_dataset, dataset_config['scalers'], config, data_type='val')
            # test set
            test_dataset.valid_indices = identify_valid_indices(test_dataset.not_nan_mask, test_dataset.sequence_length, config.config['predict_steps_ahead'])
            test_dataset.valid_timestamps = pd.to_datetime(test_dataset.timestamps[test_dataset.valid_indices.cpu().numpy()])
            evaluate_model_on_dataset(eval_folder / 'test', model, test_dataset, dataset_config['scalers'], config, data_type='test')

            print('')

        # save the split info
        with open(config.config['save_folder'] / 'split_info.pkl', 'wb') as f:
            pickle.dump(split_info, f)

        print(f"Run {config.config['experiment_name']} completed")

        #raise ValueError("Stop here")


        print("\n")






if __name__ == '__main__':
    main()
