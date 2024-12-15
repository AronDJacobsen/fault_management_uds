import os
import shutil
import random


import numpy as np
import torch
import pytorch_lightning as pl










### Helper functions



def get_repo_path(repo_name='continuous_heat_atlas'):
    # get the current path
    current_path = os.getcwd()
    
    # try to iterate back so that we find the repo until we reach the root
    while True:
        # Check if the repo_name is in the current directory
        if repo_name in os.listdir(current_path):
            # If found, construct the full path to the repo and return it
            return os.path.join(current_path, repo_name)
        
        # Move up one level to the parent directory
        parent_path = os.path.dirname(current_path)
        
        # Check if we have reached the root directory (parent_path will not change if we're at root)
        if current_path == parent_path:
            # If we're at the root, the repo was not found; return None
            return None
        
        # Update current_path to move up one level
        current_path = parent_path

# data
def data_folders(repo_path):
    """

    """
    data_path = join_path_and_str(repo_path, 'data')
    # the other
    external = join_path_and_str(data_path, 'external')
    interim = join_path_and_str(data_path, 'interim')
    processed = join_path_and_str(data_path, 'processed')
    raw = join_path_and_str(data_path, 'raw')
    return data_path, (external, interim, processed, raw)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

    # enable deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# define training parameters
def get_accelerator(verbose=True):
    # Check for CUDA availability
    if torch.cuda.is_available():
        print("Using CUDA") if verbose else None
        return 'cuda'

    # Check for MPS availability (for Apple Silicon)
    if torch.backends.mps.is_available():
        print("Using MPS") if verbose else None
        return 'mps'

    # Default to CPU if no other accelerators are available
    print("Using CPU") if verbose else None
    return 'cpu'




def folder_cleanup(experiment_dir):

    # clean up directories if they contain any .ckpt file
    if os.path.exists(experiment_dir):
        # iterate the experiment folders in the experiment directory
        for experiment in os.listdir(experiment_dir):
            experiment_path = os.path.join(experiment_dir, experiment)

            # if it is not a directory, skip
            if not os.path.isdir(experiment_path):
                continue

            # recursively iterate the folder, return True if any .ckpt file is found
            if not iterate_folder(experiment_path, '.ckpt'):
                # remove the folder
                shutil.rmtree(experiment_path)
                print(f"Removed {experiment_path}")


def iterate_folder(folder, search_for):
    """
    Recursively iterate the folder and return True if any file with the search_for extension is found.
    In case of a new folder, the function is called recursively.
    """
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(search_for):
                return True
        for dir in dirs:
            if iterate_folder(os.path.join(root, dir), search_for):
                return True
    return False



