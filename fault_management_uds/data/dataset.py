import copy

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from fault_management_uds.utilities import get_accelerator
from fault_management_uds.data.transform import CustomScaler


from fault_management_uds.data.hdf_tools import load_dataframe_from_HDF5
from fault_management_uds.data.process import remove_nans_from_start_end
from fault_management_uds.data.features import add_rain_event_priority, add_feature_engineering
from fault_management_uds.data.format import merge_intervals
from fault_management_uds.config import REFERENCE_DIR




def get_sensor_dataset(data, dataset_args, data_idx=None, scalers={}, priority_weight=None, verbose=True, dataset_type='train'):
    # prepare the data
    data = copy.deepcopy(data)

    # Ensure the data contains all required columns
    missing_columns = [col for col in dataset_args['variable_list'] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    # ensure correct column order
    data = data[dataset_args['variable_list']]


    # Filter the data if indices are provided
    if data_idx is not None:
        data = data.iloc[data_idx]

    # Extract relevant information
    columns = list(data.columns)
    timestamps = data.index
    data = data.to_numpy()
    not_nan_mask = np.logical_not(np.isnan(data))


    # scale the data
    for i, variable in enumerate(dataset_args['variable_list']):
        # scale the data
        scaler = scalers[variable]
        data[not_nan_mask[:, i], i] = scaler.transform(data[not_nan_mask[:, i], i].reshape(-1, 1)).flatten()
    
    # create the datasets
    dataset = SensorDataset(data, columns, timestamps, not_nan_mask, dataset_args, priority_weight=priority_weight, verbose=verbose, dataset_type=dataset_type)
    return dataset


def get_datasets(data, train_index, val_index, test_index, dataset_args):
    data = copy.deepcopy(data)

    # extract relevant information for the training data
    train_data = copy.deepcopy(data.iloc[train_index])
    priority_weight = train_data['priority_weight'].to_numpy()
    train_data = train_data[dataset_args['variable_list']].to_numpy()
    not_nan_mask = np.logical_not(np.isnan(train_data))

    # fit the scalers on the training data
    scalers = {}
    for i, variable in enumerate(dataset_args['variable_list']):
        # get the variable data
        var_data = train_data[not_nan_mask[:, i], i]
        # fit the scaler
        scaler = CustomScaler(
            scaler_type=dataset_args['scaler_type'], 
            # feature range must be a tuple
            feature_range=(dataset_args['feature_range'][0], dataset_args['feature_range'][1]),
            function_transform_type=dataset_args['function_transform_type'],
            obvious_min=dataset_args['obvious_min'],
            precision=dataset_args['precision'],
            )
        scaler.fit(var_data.reshape(-1, 1)) # reshape to 2D array
        scalers[variable] = scaler
    del train_data # free up memory

    # create the datasets
    train_dataset = get_sensor_dataset(data, dataset_args, train_index, scalers, priority_weight=priority_weight, dataset_type='train', verbose=True)
    val_dataset = get_sensor_dataset(data, dataset_args, val_index, scalers, dataset_type='val', verbose=False)
    test_dataset = get_sensor_dataset(data, dataset_args, test_index, scalers, dataset_type='test', verbose=False)

    dataset_config = {
        'train_index': train_index, 'val_index': val_index, 'test_index': test_index, 
        'scalers': scalers, 
        'train_timestamps': train_dataset.timestamps, 'val_timestamps': val_dataset.timestamps, 'test_timestamps': test_dataset.timestamps,
        }
    return train_dataset, val_dataset, test_dataset, dataset_config


def handle_splits(n_obs, dataset_args):
    # store the splits
    splits = []
    # split by percentage
    if dataset_args['n_splits'] == 1:
        # split by percentage, idx ordered by train, val, test
        train_end = int(n_obs * dataset_args['train_split'])
        val_end = int(n_obs * (dataset_args['train_split'] + dataset_args['val_split']))
        
        train_index = list(range(train_end))
        val_index = list(range(train_end, val_end))
        test_index = list(range(val_end, n_obs))
        
        splits.append((train_index, val_index, test_index))

    # time series split
    else:
        testing_pct = 1 - dataset_args['train_split']
        tscv = TimeSeriesSplit(n_splits=dataset_args['n_splits'], test_size=int(n_obs * testing_pct))
        for train_index, test_index in tscv.split(range(n_obs)):
            val_split = int(len(test_index) * dataset_args['val_split'])
            val_index = test_index[:val_split]
            test_index = test_index[val_split:]
            splits.append((train_index, val_index, test_index))

    return splits




class SensorDataset(Dataset):
    def __init__(self, data, columns=None, timestamps=None, not_nan_mask=None, dataset_args=None, priority_weight=None, verbose=True, dataset_type='train'):
        self.dataset_args = dataset_args
        self.dataset_type = dataset_type
        self.verbose = verbose
        self.device = get_accelerator(verbose=False)
        self.sequence_length = dataset_args.get('sequence_length', 60)
        self.steps_ahead = dataset_args.get('steps_ahead', 1)
        
        self.data = torch.tensor(data, dtype=torch.float32).to(self.device)

        # handle NaNs and identify valid indices
        self.not_nan_mask = torch.tensor(not_nan_mask, dtype=torch.bool).to(self.device)
        self.valid_indices = identify_valid_indices(self.not_nan_mask, self.sequence_length, self.steps_ahead)
        invalid = (len(self.data) - len(self.valid_indices)) - (self.sequence_length + self.steps_ahead - 1)
        print(f"Validity: {invalid} minutes are invalid.") if self.verbose else None

        # filter priority weight based on valid indices
        if priority_weight is not None:
            self.priority_weight = priority_weight[self.valid_indices.cpu().numpy()]

        # store the columns, and timestamps
        self.columns = np.array(columns)
        self.timestamps = np.array(timestamps).astype(str)
        self.valid_timestamps = pd.to_datetime(self.timestamps[self.valid_indices.cpu().numpy()])

        # store the independent and dependent variables
        self.engineered_vars = dataset_args.get('engineered_vars', [])
        self.engineered_idx = np.array([self.columns.tolist().index(var) for var in self.engineered_vars])
        self.exogenous_vars = dataset_args.get('exogenous_vars', [])
        self.exogenous_idx = np.array([self.columns.tolist().index(var) for var in self.exogenous_vars])
        self.endogenous_vars = dataset_args.get('endogenous_vars', [])
        self.endogenous_idx = np.array([self.columns.tolist().index(var) for var in self.endogenous_vars])

        self.noise_injection = dataset_args.get('noise_injection', False) if self.dataset_type == 'train' else False

        # validate data
        self._validate_data()


    def _validate_data(self):
        # Check if all data arrays (data, not_nan_mask, timestamps) have the same length
        if not (len(self.data) == len(self.not_nan_mask) == len(self.timestamps)):
            raise ValueError(f"Data, not_nan_mask, and timestamps must have the same length. Got lengths {len(self.data)}, {len(self.not_nan_mask)}, and {len(self.timestamps)}.")
        
        # Check if endogenous_idx is within bounds for each data element
        if self.endogenous_idx.size == 0:
            raise ValueError("No dependent variables specified.")
            
        # Ensure idx + sequence_length does not exceed the bounds of the data
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length {self.sequence_length} must be greater than 0.")
        
        # Ensure that there's enough data for the given sequence length
        if len(self.valid_indices) == 0:
            raise ValueError(f"No valid sequences found with sequence_length {self.sequence_length}.")
            
        print("Data validation passed.") if self.verbose else None

    def __len__(self):
        # return the number of possible indices
        return len(self.valid_indices)


    def __getitem__(self, idx):
        # Get the original index from the valid indices
        idx = self.valid_indices[idx]

        # the index is the starting point of the sequence (valid idx)
        y = self.data[idx, self.endogenous_idx]  # Dependent variable at next timestep
        mask = self.not_nan_mask[idx, self.endogenous_idx]  # Mask value
        starttime = self.timestamps[idx]  # Timestamp of the next step

        # Get the sequence of data
        x = self.data[idx - self.sequence_length:idx]      
        # Optionally, add slight noise to x (if re-enabled)
        x += torch.randn_like(x) * 1e-6 if self.noise_injection else 0
        return x, y, mask, starttime




def identify_valid_indices(not_nan_mask, sequence_length, steps_ahead=1):
    """
    Extracts valid sequences based on the specified sequence length and steps ahead.
    A valid sequence has no NaN values.
    
    Parameters:
    - not_nan_mask (Tensor): Boolean tensor where True indicates a valid value.
    - sequence_length (int): Length of the sequence window to evaluate.
    - steps_ahead (int): Additional steps to evaluate after the sequence.

    Returns:
    - valid_window_mask (Tensor): Boolean tensor where True indicates a valid window.
    """
    device = get_accelerator(verbose=False)
    # Ensure input is boolean and handle multi-dimensional data
    if not_nan_mask.ndim > 1:
        not_nan_mask = not_nan_mask.all(dim=1)
    
    # Compute rolling window length
    rolling_window_length = sequence_length + steps_ahead

    # Convolve with a kernel of ones to identify valid windows
    kernel = torch.ones(1, 1, rolling_window_length, dtype=torch.float32).to(device)
    valid_window_mask = torch.nn.functional.conv1d(
        not_nan_mask.float().view(1, 1, -1),  # Input signal
        weight=kernel,
        stride=1
    ).squeeze(0).squeeze(0) # Remove batch and channel dimensions
    # the result is: 

    # Numerical precision handling; True if all values in the window are valid
    valid_window_mask = (valid_window_mask.round() == rolling_window_length)

    # Adjust padding to align with input size
    pad_start = rolling_window_length - 1
    valid_window_mask = torch.cat([
        torch.zeros(pad_start, dtype=torch.bool).to(device),
        valid_window_mask,
    ])

    # Identify valid indices
    valid_indices = torch.where(valid_window_mask)[0]
    # Shift indices to the start of the sequence
    valid_indices -= steps_ahead - 1

    return valid_indices




def load_data(timestamps, data_file_path, dataset_args, data_type='train'):
    # load the data
    starttime, endtime = timestamps[0], timestamps[-1]
    data, _, _, _ = load_dataframe_from_HDF5(data_file_path, "combined_data/clean", columns=dataset_args['data_variables'], starttime=starttime, endtime=endtime, complete_range=True)

    # only include valid endogenous variables
    if data_type == 'complete':
        data = remove_nans_from_start_end(data, columns=dataset_args['endogenous_vars'])

    # add rain event priority
    if data_type in ['complete', 'train']:
        data = add_rain_event_priority(data, dataset_args)

    # add feature engineering
    if dataset_args['engineered_vars']:
        data = add_feature_engineering(data, dataset_args)

    return data


def load_conditions(starttime, endtime):
    overall_timestamps = pd.date_range(starttime, endtime, freq='1min')

    # rain events and extreme events
    rain_events = pd.read_csv(REFERENCE_DIR / 'events' / 'rain_events.csv')
    rain_events['start'], rain_events['end'] = pd.to_datetime(rain_events['start']), pd.to_datetime(rain_events['end'])
    # to ensure system response, add 2 hours to the end of the rain event
    rain_events['end'] = rain_events['end'] + pd.Timedelta(minutes=60*2)
    rain_events = rain_events[(rain_events['start'] >= starttime) & (rain_events['end'] <= endtime)]
    # extreme events
    extreme_events = rain_events[rain_events['extreme'] == True]
    # rain events without extreme events
    rain_events = rain_events[~rain_events.index.isin(extreme_events.index)]

    # dry conditions
    dry_conditions = pd.read_csv(REFERENCE_DIR / 'events' / 'dry_conditions.csv') 
    dry_conditions['start'], dry_conditions['end'] = pd.to_datetime(dry_conditions['start']), pd.to_datetime(dry_conditions['end'])
    dry_conditions = dry_conditions[(dry_conditions['start'] >= starttime) & (dry_conditions['end'] <= endtime)]


    if (rain_events.empty) or (extreme_events.empty) or (dry_conditions.empty):
        print("Some conditions are empty. Assigning dummy values.")
        # assign dummy values
        dummy_df = pd.DataFrame({'start': [starttime], 'end': [endtime]})
        rain_events, extreme_events, dry_conditions = dummy_df, dummy_df, dummy_df
        rain_timestamps, extreme_timestamps, dry_timestamps = overall_timestamps, overall_timestamps, overall_timestamps        

    else:
        # merge intervals if they overlap
        rain_events = merge_intervals(rain_events)
        rain_timestamps = pd.DatetimeIndex(pd.concat([
            pd.Series(pd.date_range(row['start'], row['end'], freq='1min'))
            for _, row in rain_events.iterrows()
        ]))

        # extreme
        extreme_events = merge_intervals(extreme_events)
        extreme_timestamps = pd.DatetimeIndex(pd.concat([
            pd.Series(pd.date_range(row['start'], row['end'], freq='1min'))
            for _, row in extreme_events.iterrows()
        ]))

        # filter
        # Generate all intervals
        dry_timestamps = pd.DatetimeIndex(pd.concat([
            pd.Series(pd.date_range(row['start'], row['end'], freq='1min'))
            for _, row in dry_conditions.iterrows()
        ]))


    conditions = {
        'Overall': {
            'color': 'gray',
            'df': pd.DataFrame(),
            'timestamps': overall_timestamps,
        },
        'Rain': {
            'color': 'lightskyblue',
            'df': rain_events,
            'timestamps': rain_timestamps,
        },
        'Extreme': {
            'color': 'violet',
            'df': extreme_events,
            'timestamps': extreme_timestamps,
        },
        'Dry': {
            'color': 'bisque',
            'df': dry_conditions,
            'timestamps': dry_timestamps,
        },
    }

    return conditions

    
