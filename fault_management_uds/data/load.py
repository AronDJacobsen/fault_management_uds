import os
from datetime import datetime
from datetime import timedelta
import datetime
import re
import h5py
import numpy as np
import pandas as pd

from fault_management_uds.data.HDF5_functions import load_dataframe_from_HDF5
from fault_management_uds.config import error_indicators



def import_metadata(file_path):
    metadata = pd.read_csv(file_path, sep=",")
    # convert the endtime to datetime
    metadata['StartTime'] = pd.to_datetime(metadata['StartTime'])
    metadata['EndTime'] = pd.to_datetime(metadata['EndTime'])
    return metadata




def get_event(
        sensor_name, start, end, 
        raw_rain_path=None, 
        raw_sensor_path=None, provided_processed_path=None, metadata=None, 
        data_file_path=None
    ):

    #print(f"Loading event from {start} to {end}")
    
    event_data = {}
    # raw rain files
    if raw_rain_path is not None:
        raw_5425 = load_data_period(raw_rain_path, ['5425'], start, end, filetype='pkl', fillna_with=0)
        raw_5427 = load_data_period(raw_rain_path, ['5427'], start, end, filetype='pkl', fillna_with=0)
        event_data['raw_5425'] = raw_5425 if raw_5425 is not None else pd.DataFrame(columns=['value'])
        event_data['raw_5427'] = raw_5427 if raw_5427 is not None else pd.DataFrame(columns=['value'])
        print("    Raw rain data loaded")

    # raw sensor files
    if raw_sensor_path is not None and metadata is not None:
        filenames = filenames_based_on_period(metadata, sensor_name, start, end, 'SaveName')
        raw_sensor = load_data_period(raw_sensor_path, filenames, start, end, filetype='pkl', fillna_with=np.nan)
        #raw_sensor = ensure_data_is_from_start_to_end(raw_sensor, start, end)
        event_data['raw_sensor'] = raw_sensor if raw_sensor is not None else pd.DataFrame(columns=['value'])
        print("    Raw sensor data loaded")

    # pre-cleaned data (use provided scripts)
    if provided_processed_path is not None and metadata is not None:
        filenames = filenames_based_on_period(metadata, sensor_name, start, end, 'ProvidedSaveName')
        provided_processed = load_data_period(provided_processed_path, filenames, start, end, filetype='csv', fillna_with=np.nan)
        if provided_processed is None:
            event_data['provided_processed'] = pd.DataFrame(columns=['raw_value', 'value_no_errors', 'man_remove','ffill', 'stamp','outbound', 'frozen', 'outlier', 'frozen_high', 'depth_s', 'level'], 
                                                            index=pd.DatetimeIndex([]))
        else:
            provided_processed = provided_2_full_range(provided_processed, start, end, error_indicators)
            event_data['provided_processed'] = provided_processed
        print("    Provided processed data loaded")

    # processed
    if data_file_path is not None:
        processed_5425, _, _, _ = load_dataframe_from_HDF5(data_file_path, "single_series/rain_gauge_data/5425", starttime=start, endtime=end, complete_range=True)
        processed_5427, _, _, _ = load_dataframe_from_HDF5(data_file_path, "single_series/rain_gauge_data/5427", starttime=start, endtime=end, complete_range=True)
        processed_raw_sensor, _, _, _ = load_dataframe_from_HDF5(data_file_path, f"single_series/sewer_data/{sensor_name}/raw", starttime=start, endtime=end, complete_range=True)
        processed_clean_sensor, _, _, _ = load_dataframe_from_HDF5(data_file_path, f"single_series/sewer_data/{sensor_name}/clean", starttime=start, endtime=end, complete_range=True)
        bools, _, _, _ = load_dataframe_from_HDF5(data_file_path, f"single_series/sewer_data/{sensor_name}/bools", starttime=start, endtime=end, complete_range=True, fillna_with=False)
        #indicator, _, _, _ = load_dataframe_from_HDF5(data_file_path, f"single_series/sewer_data/{sensor_name}/{sensor_name}_indicator", starttime=start, endtime=end, complete_range=True, fillna_with=0) # No data

        event_data['processed_5425'] = processed_5425
        event_data['processed_5427'] = processed_5427
        event_data['processed_raw_sensor'] = processed_raw_sensor
        event_data['processed_clean_sensor'] = processed_clean_sensor
        event_data['bools'] = bools
        #event_data['indicator'] = indicator
        print("    Processed data loaded")

    #print("Event loaded")   
    return event_data





def load_data_period(data_path, filenames, starttime, endtime, filetype='pkl', fillna_with=np.nan):
    if filenames is None:
        return None
    if isinstance(filenames, str):
        filenames = [filenames]
    # Initialize an empty list to hold the data
    all_data = []

    # Loop through all filenames (since there may be multiple now)
    for filename in filenames:
        if filetype == 'pkl':
            data = pd.read_pickle(data_path / f"{filename}.pkl")
        elif filetype == 'csv':
            data = pd.read_csv(data_path / f"{filename}.csv", index_col=0)
        
        # Convert 'time' to datetime
        data['time'] = pd.to_datetime(data['time'])
        # Sort the data
        data = data.sort_values('time')
        # Filter by the time range
        if starttime is not None and endtime is not None:
            # check if endtime has hours specified
            if endtime.hour == 0 and endtime.minute == 0 and endtime.second == 0:
                # endtime is the start of the day, add a day to include the endtime
                endtime = endtime + timedelta(days=1)
            data = data[(data['time'] >= starttime) & (data['time'] <= endtime)]
        
        # Append the data to the list if it's not empty
        if not data.empty:
            all_data.append(data)

    if not all_data:
        print(f"    No data found for period {starttime} to {endtime}")
        return None

    # Concatenate all dataframes into one
    all_data = pd.concat(all_data)

    # Set 'time' as the index
    all_data = all_data.set_index('time')

    # Fill NaNs with the specified value
    all_data = all_data.fillna(fillna_with)

    return all_data



# Loading metadata
def import_external_metadata(scripts_path):
    """Load metadata, ie. the on sensors and manual removals."""

    metadata = pd.read_csv(scripts_path / '#9_Scripts' / 'etc' / 'obs_input.csv', sep=";")
    # convert the endtime to datetime
    metadata['StartTime'] = pd.to_datetime(metadata['StartTime'], format='%d-%m-%Y')
    metadata['EndTime'] = pd.to_datetime(metadata['EndTime'], format='%d-%m-%Y')
    sort_cols = ['IdMeasurement', 'Source', 'StartTime']
    # apply the sort
    metadata = metadata.sort_values(by=sort_cols)

    manual_remove = pd.read_csv(scripts_path / '#9_Scripts' / 'etc' / 'manual_remove.csv', sep=";")

    return metadata, manual_remove



def filenames_based_on_period(metadata, sensor_name, start, end, savename):
    # Floor start and ceil end to day to cover the entire period
    # check if start and end time is none
    if start is None:
        start = metadata['StartTime'].min()
    if end is None:
        end = metadata['EndTime'].max()


    # check if endtime has hours specified
    if end.hour == 0 and end.minute == 0 and end.second == 0:
        # endtime is the start of the day, add a day to include the endtime
        end = end + timedelta(days=1)

    start = start.floor('D')
    end = end.ceil('D')


    # filter out nan savenames
    metadata = metadata[metadata[savename].notnull()]

    # Get rows for the specific sensor
    sensor_rows = metadata[metadata['IdMeasurement'] == sensor_name]

    # Filter rows that overlap with the desired period
    sensor_rows = sensor_rows[(sensor_rows['StartTime'] <= end) & (sensor_rows['EndTime'] >= start)]

    if sensor_rows.empty:
        print(f"    No data found for sensor {sensor_name} and period {start} to {end}")
        return None
    
    # Get the list of filenames that cover this period
    filenames = sensor_rows[savename].values

    # if a single filename is returned, convert it to a list
    if isinstance(filenames, str):
        filenames = [filenames]
    

    return filenames


def provided_2_full_range(sensor_data, start, end, error_indicators):

    # check if start and end time is none
    if start is None:
        start = sensor_data.index.min()
    if end is None:
        end = sensor_data.index.max()
    # create a full range indicator
    full_time_range = pd.date_range(start=start, end=end, freq='1min')
    # if sensor data is empty, create an empty dataframe
    if sensor_data is None:
        # create an empty dataframe with the columns of the other full in the full time range
        sensor_data = pd.DataFrame(index=full_time_range, columns=error_indicators + ['raw_value', 'value_no_errors', 'depth_s'])
        sensor_data[error_indicators] = False
        return sensor_data

    # reindex the sensor data to the full time range
    # sort values by depth_s
    sensor_data = sensor_data.sort_values('depth_s')
    # drop duplicates
    sensor_data = sensor_data[~sensor_data.index.duplicated(keep='first')]
    sensor_data = sensor_data.reindex(full_time_range, fill_value=np.nan)
    sensor_data[error_indicators] = sensor_data[error_indicators].fillna(False)

    return sensor_data



