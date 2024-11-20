
import h5py
import pandas as pd
import numpy as np
import os
import anytree
from anytree.exporter import DotExporter
from anytree import Node, RenderTree
import h5py
import nexusformat.nexus as nx
from datetime import timedelta




# Function to recursively build a tree with anytree from a NeXus tree
def build_tree(nexus_node, parent=None):
    tree_node = Node(str(nexus_node.nxname), parent=parent)
    if isinstance(nexus_node, nx.NXgroup):
        for child in nexus_node:
            build_tree(nexus_node[child], parent=tree_node)
    return tree_node


def print_tree(data_file_path, group=None, print_tree=True, save_path=None):
    # Load the NeXus file
    f = nx.nxload(data_file_path)
    if group is not None:
        f = f[group]

    if print_tree:
        # Build and render the tree
        root = build_tree(f)
        for pre, fill, node in RenderTree(root):
            print(f"{pre}{node.name}")
    
    if save_path is not None:
        save_path = str(save_path)
        with open(str(save_path) + ".txt", "w", encoding="utf-8") as f:
            f.write(" ")
            for pre, fill, node in RenderTree(root):
                f.write(f"{pre}{node.name}\n")
            f.write(" ")
    return f


def delete_group(data_file_path, group, verbose=True):
    with h5py.File(data_file_path, 'a') as hdf:
        if group in hdf:
            del hdf[group]
            print(f"Group '{group}' deleted") if verbose else None
        else:
            print(f"Group '{group}' does not exist") if verbose else None


def create_group(data_file_path, group, verbose=True):
    # create a group if it does not exist
    nested_group = group.split('/')
    for i in range(1, len(nested_group)+1):
        group_path = '/'.join(nested_group[:i])
        #print(f"Creating group '{group_path}'")
        with h5py.File(data_file_path, 'a') as hdf:
            if group_path not in hdf:
                hdf.create_group(group_path)
                print(f"    Group '{group_path}' created") if verbose else None
            else:
                print(f"    Group '{group_path}' already exists") if verbose else None



def overwrite_dataset(data_file_path, group_path, dataset_name, data, verbose=True):
    # Open the HDF5 file in read/write mode
    with h5py.File(data_file_path, 'a') as hdf_file:
        # Create or access the group
        group = hdf_file.require_group(group_path)
        
        # Check if the dataset exists
        if dataset_name in group:
            # If it exists, delete it
            del group[dataset_name]
        
        # Now create the dataset
        group.create_dataset(dataset_name, data=data)
        print(f"    Dataset '{dataset_name}' overwritten in group '{group.name}'") if verbose else None



def save_dataframe_in_HDF5(data_file_path, group_path, data_name, df):
    # HDF storage
    df = df.copy()

    # handle data for the HDF5 file
    df['time'] = df['time'].astype('int64')
    
    # create the data files
    timestamps = df['time'].to_numpy()
    df = df.drop(columns='time')
    data = df.to_numpy()
    columns = df.columns.to_numpy().astype('S')

    # create the group
    data_path = f"{group_path}/{data_name}"
    create_group(data_file_path, data_path, verbose=False)

    # save the data
    overwrite_dataset(data_file_path, data_path, "data", data, verbose=False)
    overwrite_dataset(data_file_path, data_path, "timestamps", timestamps, verbose=False)
    overwrite_dataset(data_file_path, data_path, "columns", columns, verbose=False)



def load_dataframe_from_HDF5(data_file_path, group_path, starttime=None, endtime=None, columns=None, complete_range=False, fillna_with=np.nan, verbose=True):
    try:
        with h5py.File(data_file_path, 'r') as hdf_file:
            #print(f"    Loading data from group: {group_path}")
            # Access the group
            group = hdf_file[group_path]
            
            # Load column names and timestamps
            all_columns = group['columns'][:].astype(str)
            all_timestamps = pd.to_datetime(group['timestamps'][:])

            # Validate columns if specified
            if columns is not None:
                # Ensure that all requested columns are valid
                invalid_cols = [col for col in columns if col not in all_columns]
                if invalid_cols:
                    raise ValueError(f"Invalid columns: {invalid_cols}. Available columns: {all_columns}")
                # Get the indices of the selected columns
                column_indices = [np.where(all_columns == col)[0][0] for col in columns]
            else:
                column_indices = slice(None)  # Select all columns
            
            # Handle starttime and endtime filtering
            if starttime is not None:
                start_idx = np.searchsorted(all_timestamps, starttime, side='left')
                if start_idx >= len(all_timestamps) or all_timestamps[start_idx] != starttime:
                    print(f"    Warning: The start time {starttime} is not in the data. The closest timestamp is {all_timestamps[start_idx]}") if verbose else None
            else:
                start_idx = 0  # Start from the beginning
                #starttime = all_timestamps[start_idx]

            if endtime is not None:
                # # check if endtime has hours specified
                # if endtime.hour == 0 and endtime.minute == 0 and endtime.second == 0:
                #     # endtime is the start of the day, add a day to include the endtime
                #     endtime = endtime + timedelta(days=1)
                end_idx = np.searchsorted(all_timestamps, endtime, side='right')
                if end_idx == 0 or all_timestamps[end_idx - 1] != endtime:
                    print(f"    Warning: The end time {endtime} is not in the data. The closest timestamp is {all_timestamps[end_idx - 1]}") if verbose else None
            else:
                end_idx = len(all_timestamps)  # Go until the end
                #endtime = all_timestamps[end_idx - 1]

            # Sanity check: Ensure valid time range
            if start_idx > end_idx:
                raise ValueError("Invalid time range: 'starttime' is greater than or equal to 'endtime'.")

            # Check if the start and end indices are the same - i.e. no data
            if start_idx == end_idx:
                # return empty DataFrame
                df = pd.DataFrame(columns=all_columns[column_indices])
                if complete_range and starttime is not None and endtime is not None:
                    # Create the full 1-minute time index from starttime to endtime
                    full_time_range = pd.date_range(start=starttime, end=endtime, freq='1min')
                    # Reindex the DataFrame to ensure 1-minute intervals across the whole time range
                    df = df.reindex(full_time_range, fill_value=fillna_with)
                return df, None, None, None

            # Load the data
            data = group['data'][start_idx:end_idx, column_indices]
            data_timestamps = all_timestamps[start_idx:end_idx]
            
            # Create the DataFrame directly using the filtered timestamps
            filtered_columns = all_columns[column_indices] if columns is None else columns
            df = pd.DataFrame(data, index=data_timestamps, columns=filtered_columns)
            df.index.name = 'time'

            if complete_range and starttime is not None and endtime is not None:
                # Create the full 1-minute time index from starttime to endtime
                full_time_range = pd.date_range(start=starttime, end=endtime, freq='1min')
                # Reindex the DataFrame to ensure 1-minute intervals across the whole time range
                df = df.reindex(full_time_range, fill_value=fillna_with)
                #print(f"Reindexed the DataFrame to the full time range from {starttime} to {endtime}")
                start_idx, end_idx, column_indices = None, None, None
            
            # order by specified columns
            if columns is not None:
                df = df[columns]

            return df, start_idx, end_idx, column_indices

    except KeyError as e:
        raise KeyError(f"Group or dataset not found in the HDF5 file: {e}")
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading data: {e}")






def update_filtered_data_in_HDF5(data_file_path, group_path, df, start_idx, end_idx, column_indices):
    # Save the data based on the filtered indices
    with h5py.File(data_file_path, 'a') as hdf_file:
        # Create or access the group
        group = hdf_file.require_group(group_path)
        dataset = group['data']
        # Save the data
        dataset[start_idx:end_idx, column_indices] = df.to_numpy()


        print(f"        Data saved in group '{group.name}'")








### OLD FUNCTIONS ###



def save_metadata(metadata, data_file_path, group_path):
    # save sensor metadata in the H5 file
    # convert object to string

    # convert to json
    metadata_json = metadata.to_json(orient='records')
    # save the metadata
    with h5py.File(data_file_path, 'a') as hdf_file:
        # Create or access the group
        group = hdf_file.require_group(group_path)
        # save the metadata, overwrite if it exists
        group.attrs['metadata'] = metadata_json
        print(f"    Metadata saved in group '{group.name}'")



def load_metadata(data_file_path, group_path):
    # load sensor metadata from the H5 file
    with h5py.File(data_file_path, 'r') as hdf_file:
        # Create or access the group
        group = hdf_file.require_group(group_path)
        # load the metadata
        metadata_json = group.attrs['metadata']
        metadata = pd.read_json(metadata_json)

        # convert the start and end time to datetime
        metadata['starttime'] = pd.to_datetime(metadata['starttime'])
        metadata['endtime'] = pd.to_datetime(metadata['endtime'])
        print(f"    Metadata loaded from group '{group.name}'")

    return metadata


