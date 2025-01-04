
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


from torch.utils.data import DataLoader


from fault_management_uds.config import get_additional_configurations
from fault_management_uds.utilities import seed_everything
from fault_management_uds.data.dataset import load_data
from fault_management_uds.modelling.models import get_model, load_model_checkpoint
from fault_management_uds.data.dataset import load_data, get_sensor_dataset
from fault_management_uds.data.hdf_tools import load_dataframe_from_HDF5
from fault_management_uds.modelling.predict import inverse_transform


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
    parser.add_argument('--data_type', type=str, default='test', help='Data type to evaluate on', choices=['train', 'val', 'test'])
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



def get_outputs(model, dataset, scalers, config):


    ### Prepare the results DataFrame

    # get the defaults: starttime, valid_index, data_label, target
    column_2_idx = {column: [i] for i, column in enumerate(['starttime', 'valid_index', 'data label', 'target', 'residual'])}

    valid_timestamps = dataset.valid_timestamps.to_numpy()
    valid_indices = dataset.valid_indices.cpu().numpy()
    # load the data label
    data_labels, _, _, _ = load_dataframe_from_HDF5(config['dataset_args']['data_file_path'], f"combined_data/{config['dataset_args']['data_group']}",
                                             columns=['Data Label'], starttime=valid_timestamps[0], endtime=valid_timestamps[-1],
                                             complete_range=True)
    # filter the data labels based on valid timestamps
    data_labels = data_labels.loc[valid_timestamps].values
    # ensure same length
    assert len(data_labels) == len(valid_timestamps), f"data labels length: {len(data_labels)}, Valid timestamps length: {len(valid_timestamps)}"

    # Handle the model outputs
    for return_name, return_dim in model.model.returns.items():
        # get the increment index
        start_idx = np.max(np.concatenate(list(column_2_idx.values()))) + 1
        # add the return columns
        column_2_idx[return_name] = list(range(start_idx, start_idx + return_dim))
    
    # Handle additional extracted features
    IG_seq_dims = 20
    IG_var_idxs = model.model.endogenous_idx # [slice(None), model.model.endogenous_idx]
    IG_var_dims = config['model_args']['input_size'] if IG_var_idxs == slice(None) else len(IG_var_idxs)
    # PIG only benefits from the 1st sequence dimension
    PIG_seq_dims = 2
    PIG_var_idxs = [model.model.endogenous_idx] # [slice(None), [model.model.endogenous_idx]]
    PIG_var_dims = config['model_args']['input_size'] if PIG_var_idxs == slice(None) else len(PIG_var_idxs)
    additional_features = {
        'IG': IG_seq_dims * IG_var_dims,
        'PIG': PIG_seq_dims * PIG_var_dims
    }
    for return_name, return_dim in additional_features.items():
        # get the increment index
        start_idx = np.max(np.concatenate(list(column_2_idx.values()))) + 1
        # add the return columns
        column_2_idx[return_name] = list(range(start_idx, start_idx + return_dim))


    # Prepare a combined numpy array to store the results
    n_col_idx = np.max(np.concatenate(list(column_2_idx.values()))) + 1
    results = np.empty((len(valid_indices), n_col_idx), dtype=object)

    # add the defaults
    results[:, column_2_idx['starttime']] = valid_timestamps.reshape(-1, 1)
    results[:, column_2_idx['valid_index']] = valid_indices.reshape(-1, 1)
    results[:, column_2_idx['data label']] = data_labels.reshape(-1, 1)


    ### Get the outputs

    # Prepare DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['training_args']['batch_size'], 
        shuffle=False, 
        num_workers=0
    )

    ig = IntegratedGradients(model)
    
    # No gradient calculation during prediction
    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader, desc='Predicting'):
            x, y, _, valid_idx = batch
            batch_size = x.size(0)

            # Forward pass
            outputs = model.model.forward(x)
            prediction = outputs[0]

            # Store the results
            results[start:start+batch_size, column_2_idx['target']] = y.cpu().numpy()

            # add the output to the results
            i_out = 0
            for return_name, return_idxs in column_2_idx.items():
                if return_name in model.model.returns:
                    output = outputs[i_out]
                    results[start:start+batch_size, return_idxs] = output.cpu().numpy()
                    i_out += 1

            # Calculate the integrated gradients
            x.requires_grad = True
            attributions = ig.attribute(x, target=0)[:, -IG_seq_dims:, IG_var_idxs].reshape(batch_size, -1)
            results[start:start+batch_size, column_2_idx['IG']] = attributions.cpu().numpy()


            # Calculate the locally integrated gradients
            actual_next_step = dataset.data[valid_idx] # construct the input (actual data)
            input = torch.cat([x[:, 1:, :], actual_next_step.unsqueeze(1)], dim=1)
            expected_next_step = actual_next_step.clone() # construct the baseline (expected by the model)
            expected_next_step[:, model.model.endogenous_idx] = prediction
            baseline = torch.cat([x[:, 1:, :], expected_next_step.unsqueeze(1)], dim=1)
            attributions = ig.attribute(input, baseline, target=0)[:, -PIG_seq_dims:, PIG_var_idxs].reshape(batch_size, -1)
            results[start:start+batch_size, column_2_idx['PIG']] = attributions.cpu().numpy()

            # Update start index
            start += batch_size


    # Inverse transform the data
    results[:, column_2_idx['target']] = inverse_transform(results[:, column_2_idx['target']], scalers, dataset.endogenous_vars, None)
    results[:, column_2_idx['prediction']] = inverse_transform(results[:, column_2_idx['prediction']], scalers, dataset.endogenous_vars, None)

    # Calculate the residuals
    results[:, column_2_idx['residual']] = results[:, column_2_idx['target']] - results[:, column_2_idx['prediction']]

    return results, column_2_idx




def main():
    # Parse arguments
    args = parse_args()

    model_save_path = args.model_save_path
    data_type = args.data_type
    data_group = args.data_group
    fast_run = args.fast_run
    num_workers = args.num_workers


    # handle fast run
    subset = 5000 if fast_run else None
    

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


    # load the model
    additional_configurations = get_additional_configurations(dataset)
    model_to_load = config['training_args']['model_to_load']
    model = load_model_checkpoint(run_folder, run_info, model_to_load, config, additional_configurations)


    # get the outputs
    outputs, column_2_idx = get_outputs(model, dataset, scalers, config)


    # save the results as pickle
    with open(outputs_folder / 'outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)
    # save the column mapping as json
    with open(outputs_folder / 'column_2_idx.json', 'w') as f:
        json.dump(column_2_idx, f)




if __name__ == '__main__':
    main()






















