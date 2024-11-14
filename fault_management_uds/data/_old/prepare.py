
import pandas as pd
from pathlib import Path
import numpy as np
import h5py
import re
from datetime import datetime
from datetime import timedelta
import datetime
import os


from fault_management_uds.config import error_indicators


def add_true_one_back(series):
    # add a true value one back, for the forward fill
    # it doesn't work if only the value to be filled is true
    for i in range(1, len(series)): # start from 1
        # if true, set the previous to true
        if series[i]:
            series[i-1] = True
    return series



def fix_columns(sensor_data, sensor_metadata, error_indicators):
    # # get the type
    # sensor_type = sensor_metadata['type'].values[0].lower()
    
    ### Adding the zero point to the sensor data
    # # get the zero point
    # zero_point = sensor_metadata['zeropoint'].values[0]
    # # fix the zero point
    # if sensor_type in ['level', 'position']:
    #     sensor_data[value_columns] = sensor_data[value_columns] + zero_point

    ### Fixing the forward fill
    sensor_data["ffill"] = add_true_one_back(sensor_data["ffill"].tolist())
    # Forward-fill only the rows indicated by the ffill column
    sensor_data.loc[sensor_data["ffill"], "raw_value"] = sensor_data["raw_value"].ffill()
    sensor_data.loc[sensor_data["ffill"], "value_no_errors"] = sensor_data["value_no_errors"].ffill()
    # drop the ffill column
    sensor_data = sensor_data.drop(columns="ffill")

    # have to reset the value_no_errors based on the error indicators
    #     value_no_errors[stamp_id | outbound_id | frozen_id2 | man_remove_id | outlier_id] = np.nan  
    error_mask = sensor_data[error_indicators].sum(axis=1) > 0
    sensor_data.loc[error_mask, "value_no_errors"] = np.nan


        
    return sensor_data




def prepare_sensor_series(sensor, sensor_2_files, file_2_metadata, rename_dict, processed_path):

    # imported

    # complete series with all the data
    complete_series = pd.DataFrame()
    value_columns = list(rename_dict.values())

    # collect all the sensor data
    for i, filename in enumerate(sensor_2_files[sensor]):
        print(f"    ({i+1}/{len(sensor_2_files[sensor])}) Loading data from file: {filename}")

        # load the data
        sensor_data = pd.read_csv(processed_path / filename, sep=",", header=0, index_col=0)

        # all errors should be in the columns, add in case missing and fill na with False
        for error in error_indicators:
            if error not in sensor_data.columns:
                sensor_data[error] = False
        sensor_data[error_indicators] = sensor_data[error_indicators].fillna(False)

        # # get the metadata
        sensor_metadata = file_2_metadata[filename]
        # fix the columns
        sensor_data = fix_columns(sensor_data, sensor_metadata, error_indicators)

        # rename the columns
        sensor_data.rename(columns=rename_dict, inplace=True)


        # combine the data
        complete_series = pd.concat([complete_series, sensor_data], axis=0)

    # post processing
    # drop level
    if 'level' in complete_series.columns:
        complete_series = complete_series.drop(columns='level')


    # Fix the time column
    # set time to datetime
    complete_series['time'] = pd.to_datetime(complete_series['time'])
    # drop duplicates by time
    complete_series = complete_series.drop_duplicates(subset='time', keep='first')
    # From the min to max time, add 1 minute intervals
    complete_series = complete_series.set_index('time').resample('1T').asfreq().reset_index() #

    # Handling nan values
    # add nan indicator columns (0 if not nan, 1 if nan), where nan means that the value is missing
    complete_series['no_data'] = complete_series['raw'].isna().astype(int)

    # Handling error values
    # fill na with 0
    complete_series[error_indicators] = complete_series[error_indicators].fillna(0)
    #error_indicators = [error for error in error_indicators if error in list(complete_series.columns)]
    complete_series['is_error'] = (complete_series[error_indicators].sum(axis=1) > 0).astype(int)

    # errors should be 0 or 1
    complete_series[error_indicators] = complete_series[error_indicators].astype(int)


    # then set the nan values to 0
    complete_series[value_columns] = complete_series[value_columns].fillna(0)
    first_columns = ['time'] + value_columns + ['no_data', 'is_error']
    other_columns = [other for other in list(complete_series.columns) if other not in first_columns]
    # for the other columns, set nan values to False and convert to int
    complete_series[other_columns] = complete_series[other_columns].fillna(False).astype(int)

    # Final adjustments
    # reorder columns
    complete_series = complete_series[first_columns + other_columns]
    # sort by time columns
    complete_series = complete_series.sort_values(by='time')
    # reset index
    complete_series = complete_series.reset_index(drop=True)
    return complete_series



