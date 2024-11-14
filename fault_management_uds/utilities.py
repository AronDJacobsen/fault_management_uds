import os



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





