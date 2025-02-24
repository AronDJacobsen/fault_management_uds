
import argparse
import os
import shutil
import json
import time
import pickle
import yaml


from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


from torch.utils.data import DataLoader


from fault_management_uds.config import get_additional_configurations
from fault_management_uds.utilities import seed_everything
from fault_management_uds.data.dataset import load_data
from fault_management_uds.modelling.models import get_model, load_model_checkpoint
from fault_management_uds.data.dataset import load_data, get_sensor_dataset
from fault_management_uds.data.hdf_tools import load_dataframe_from_HDF5
from fault_management_uds.modelling.predict import inverse_transform

from fault_management_uds.train_model import handle_anomalous_iteration

from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR


# Set default global settings for tqdm
tqdm_kwargs = {'mininterval': 60.0}  # Updates every _ seconds
# Replace the tqdm function globally with pre-configured options
import builtins
builtins.tqdm = lambda *args, **kwargs: tqdm(*args, **{**tqdm_kwargs, **kwargs})



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--model_save_path', type=str, default='transformer/7_anomalous/1_iteration', help='Folder where the model is saved')
    #parser.add_argument('--data_types', type=list, default=['test'], help='Select data types to run')
    parser.add_argument("--data_types", nargs='+', required=True, help="Data types (e.g., train, val, test).")
    parser.add_argument('--data_group', type=str, default='anomalous', help='Data group to evaluate on', choices=['clean', 'anomalous'])
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()


def load_run_info(experiment_folder, data_type, subset=None):
    save_folder = MODELS_DIR / experiment_folder

    split = 0 # first split
    relative_path = f'{split+1}_split/'
    run_folder = save_folder / relative_path
    print(save_folder)

    # load the config
    config = yaml.load(open(save_folder / 'config.yaml', 'r'), Loader=yaml.Loader)
    print(f"Total runs: {config['dataset_args']['n_splits']}") 
    seed = config['training_args']['seed']
    seed_everything(seed)

    # run info for each split
    split_info = torch.load(save_folder / 'split_info.pkl', map_location='cpu')
    run_info = split_info[split]    
    # adjust the paths if needed
    if '/' in run_info['best_model_path']:
        run_info['best_model_path'] = os.path.relpath(run_info['best_model_path'], run_folder)
        run_info['last_model_path'] = os.path.relpath(run_info['last_model_path'], run_folder)
        run_info['top_k_best_model_paths'] = {
            k: os.path.relpath(str(v), run_folder) for k, v in run_info['top_k_best_model_paths'].items()
        }

    # load the data
    scalers = run_info['dataset_config']['scalers']

    # load the data
    data_indices = run_info['dataset_config'][f'{data_type}_index']
    print(f"Data type: {data_type}, Data indices: {len(data_indices)}")

    # Subset if specified
    data_indices = data_indices if subset is None else data_indices[:subset]

    return config, run_folder, data_indices, scalers, run_info


# def apply_pca(ig_results, pca=None, scaler=None, explain_variance=0.95):
#     # if none, then fit
#     if pca is None:
#         # Standardize the data
#         scaler = StandardScaler()
#         # fit the scaler
#         scaler.fit(ig_results)
#         # Apply PCA to the IG results
#         pca = PCA(n_components=explain_variance, svd_solver='full')
#         # Fit the PCA
#         pca.fit(ig_results)
#     else:
#         print("Using the provided PCA")
    
#     # Scale the data
#     ig_results = scaler.transform(ig_results)

#     # Transform the data
#     ig_results_pca = pca.transform(ig_results)
#     print(f"Original IG shape: {ig_results.shape}, PCA IG shape: {ig_results_pca.shape}")
#     print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")
#     print(f"N components: {pca.n_components_}")
#     return ig_results_pca, pca, scaler

def apply_pca(ig_results, pca=None, scaler=None, explain_variance=0.95, batch_size=1000):
    if pca is None:
        scaler = StandardScaler()
        scaler.fit(ig_results)
        ig_results_scaled = scaler.transform(ig_results)
        
        # Use Incremental PCA instead of standard PCA
        pca = IncrementalPCA(n_components=explain_variance, batch_size=batch_size)
        
        for i in range(0, ig_results_scaled.shape[0], batch_size):
            batch = ig_results_scaled[i:i + batch_size]
            pca.partial_fit(batch)  # Fit incrementally
        # Indicate variance explained
        print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")
        
    else:
        print("Using the provided PCA")
        ig_results_scaled = scaler.transform(ig_results)

    ig_results_pca = pca.transform(ig_results_scaled)
    print(f"Original IG shape: {ig_results.shape}, PCA IG shape: {ig_results_pca.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()}")
    print(f"N components: {pca.n_components_}")
    
    return ig_results_pca, pca, scaler



def features(model, dataset, scalers, config, pca, scaler, num_workers=0):


    ### Prepare the results DataFrame

    # get the defaults: starttime, valid_index, data_label, target

    valid_timestamps = dataset.valid_timestamps.to_numpy()
    valid_indices = dataset.valid_indices.cpu().numpy()

    # Handle additional extracted features
    # IG_seq_dims = 20
    # IG_var_idxs = model.model.endogenous_idx # [slice(None), model.model.endogenous_idx]
    # IG_var_dims = config['model_args']['input_size'] if IG_var_idxs == slice(None) else len(IG_var_idxs)
    
    IG_seq_dims = config["dataset_args"]["sequence_length"] # use the complete sequence
    IG_var_dims = config['model_args']['input_size']
    IG_return_dim = IG_seq_dims * IG_var_dims
    # Define the IG to store the results
    ig_results = np.empty((len(valid_indices), IG_return_dim), dtype=object)
    print(f"IG results shape: {ig_results.shape}")


    ### Get the outputs
    # Prepare DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training_args']['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
    )

    ig = IntegratedGradients(model)
    
    # No gradient calculation during prediction
    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader, desc='Predicting'):
            x, y, _, valid_idx = batch
            batch_size = x.size(0)

            # Forward pass
            #outputs = model.model.forward(x)

            # Integrated gradients
            x.requires_grad = True
            attributions = ig.attribute(x, target=0)[:, :, :]
            # reshape the attributions
            attributions = attributions.reshape(batch_size, IG_return_dim)
            # Store the attributions
            ig_results[start:start+batch_size] = attributions.cpu().numpy()
            start += batch_size

    # Convert to numpy array
    #ig_results = np.array(ig_results, dtype=float)
    ig_results = np.array(ig_results, dtype=np.float32)
    
    # # print and example
    # print(f"IG results shape: {ig_results.shape}")
    # print(f"Example: {ig_results[0]}")

    # # check if nans
    # if np.isnan(ig_results).any():
    #     print("There are NaNs in the IG results")
    #     # indicate where the nans are in axis 0
    #     nan_idxs = np.isnan(ig_results).any(axis=1)
    #     print(f"Number of NaNs: {nan_idxs.sum()}")
    #     print(f"Indices: {nan_idxs}")
    #     print(f"Indices of NaNs: {valid_indices[nan_idxs]}")

    # Reduce IG dims with PCA
    #explain_variance = 0.95
    explain_variance = 20 # use 10 components
    ig_results, pca, scaler = apply_pca(ig_results, pca=pca, scaler=scaler, explain_variance=explain_variance, batch_size=10000)

    return ig_results, pca, scaler




def main():
    # Parse arguments
    args = parse_args()

    model_save_path = args.model_save_path
    data_types = args.data_types
    data_group = args.data_group
    fast_run = args.fast_run
    num_workers = args.num_workers

    # PCA
    pca = None # starts with none, until 'train' data is processed, then it is used for the rest of the data
    scaler = None

    # handle fast run
    subset = 10000 if fast_run else None
    
    # Ensure data types are in the correct order
    dt_order = ['train', 'val', 'test']
    data_types = [dt for dt in dt_order if dt in data_types]
    print(f"Selected data types: {data_types}")

    # iterate the selected datatypes
    for data_type in data_types:

        # load run info
        config, run_folder, data_indices, scalers, run_info = load_run_info(model_save_path, data_type, subset=subset)

        # define save folder
        outputs_folder = run_folder / 'anomalous'
        outputs_folder.mkdir(parents=True, exist_ok=True)

        # Load data
        data = load_data([None, None], config['dataset_args']['data_file_path'], config['dataset_args'], 
                        data_type=data_type,
                        data_group=data_group,
                        )
        dataset = get_sensor_dataset(data, config['dataset_args'], data_indices, scalers, dataset_type=data_type, verbose=False)
        del data

        # handle anomalous iteration if specified
        if config['dataset_args'].get('anomalous_iteration', None) is not None:
            print(f"Handling anomalous iteration for {data_type} data")
            dataset = handle_anomalous_iteration(config, config['training_args']['fine_tune_path'], dataset, data_type)


        # load the model
        additional_configurations = get_additional_configurations(dataset)
        model_to_load = config['training_args']['model_to_load']
        model = load_model_checkpoint(run_folder, run_info, model_to_load, config, additional_configurations)


        # Get the outputs
        if pca is None:
            print(f"### Fitting PCA for {data_type} data ###")
        else:
            print(f"### Using a fitted PCA for {data_type} data ###")
        
        ig_results, pca, scaler = features(model, dataset, scalers, config, pca, scaler, num_workers=num_workers)



        results_save_path = outputs_folder / data_type
        results_save_path.mkdir(parents=True, exist_ok=True)

        # save the results as pickle
        with open(results_save_path / f'ig_results.pkl', 'wb') as f:
            pickle.dump(ig_results, f)


        print(f"Finished processing {data_type} data")
        print(f"Outputs saved at: {results_save_path}")
        print("-"*50); print("\n")



if __name__ == '__main__':
    main()










