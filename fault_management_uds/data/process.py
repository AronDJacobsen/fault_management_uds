
import datetime
from datetime import timedelta
import re

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from fault_management_uds.data.bellinge import handle_ifix_data_saving

############################################################################################################
# Improved processing
############################################################################################################


def consecutive_nans_until_not(lst, consecutive_count=5):
    """
    Returns a list of booleans indicating if the corresponding value in the input list is part of a streak of NaNs.
    """
    

    result = []
    nan_streak = False  # Flag to mark if we are in a streak of NaNs
    streak_count = 0     # Counter for consecutive NaNs

    for val in lst:
        if np.isnan(val):
            streak_count += 1  # Count consecutive NaNs
        else:
            # If we encounter a non-NaN and there's a streak of 5 or more NaNs
            if streak_count >= consecutive_count:
                result.extend([True] * streak_count)
            else:
                result.extend([False] * streak_count)
            result.append(False)  # For the current non-NaN value
            streak_count = 0  # Reset the streak count

    # If the last value is NaN
    if streak_count >= consecutive_count:
        result.extend([True] * streak_count)
    else:
        result.extend([False] * streak_count)

    return result



def remove_nans_from_start_end(df, column_name):
    df = df.copy()
    # check if time is in the index and set it as a column instead
    time_is_index = False
    if isinstance(df.index, pd.DatetimeIndex):
        time_is_index = True
        df.reset_index(inplace=True, drop=False, names='time')

    df.sort_values(by='time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    # Find the first non-NaN index for 'raw_value'
    first_non_nan_index = df[column_name].notna().idxmax()
    
    # Find the last non-NaN index for 'raw_value' by reversing the series
    last_non_nan_index = df[column_name][::-1].notna().idxmax()

    # Slice the DataFrame to remove NaNs from the start and end
    sliced_df = df.iloc[first_non_nan_index:last_non_nan_index + 1]
    sliced_df.reset_index(drop=True, inplace=True)

    # check if time was in the index and set it back as the index
    if time_is_index:
        sliced_df.set_index('time', inplace=True)

    return sliced_df


def ensure_data_is_from_start_to_end(data, start, end):
    """
    Ensures that the given data has a DatetimeIndex and fills in missing time points with NaN.

    Args:
        data (pd.Series or pd.DataFrame): The input data with a DatetimeIndex.
        start (pd.Timestamp): The start time.
        end (pd.Timestamp): The end time.

    Returns:
        pd.DataFrame or pd.Series: The data with a complete DatetimeIndex and missing values filled with NaN.

    Raises:
        ValueError: If the data does not have a DatetimeIndex, if start is after end, 
                    or if the reindexed data is empty.
    """

    # Check if data is a Series or DataFrame
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise ValueError("Input data must be a pandas Series or DataFrame.")

    # Ensure the data has a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data must have a DatetimeIndex.")
    
    # Ensure start is before or equal to end
    if start > end:
        raise ValueError("Start time must be before or equal to end time.")
    
    # Remove duplicate indices, averaging values if necessary
    if data.index.duplicated().any():
        data = data.groupby(data.index).mean()

    # check if endtime has hours specified
    if end.hour == 0 and end.minute == 0 and end.second == 0:
        # endtime is the start of the day, add a day to include the endtime
        end = end + timedelta(days=1)

    # Generate a complete time range from start to end with 1-minute frequency
    idx = pd.date_range(start=start, end=end, freq='1min')

    # Reindex the data to the new index, filling missing times with NaN
    data_reindexed = data.reindex(idx, fill_value=np.nan)

    # If the reindexed data is empty, raise an error
    if data_reindexed.empty:
        raise ValueError("Reindexed data is empty, please check the time range and data.")

    # # Handle boolean columns by interpolation
    # for bool_col in bool_columns:
    #     if bool_col == 'ffill':
    #         # fill the NaN values with 1 and convert to bool
    #         data_reindexed[bool_col] = data_reindexed[bool_col].fillna(1)
    #     else:
    #         # convert to int
    #         data_reindexed[bool_col] = data_reindexed[bool_col].astype(float)
    #         # interpolate
    #         data_reindexed[bool_col] = data_reindexed[bool_col].interpolate(method='linear')
    #     # for values larger than 0, set to True
    #     data_reindexed[bool_col] = data_reindexed[bool_col] > 0
    return data_reindexed




def resample_data(data):
    columns = data.columns.tolist()


    # resample the data
    resampled_data = data.copy().resample('1min').mean()

    # # handle ffill
    resampled_data['ffill'] = resampled_data['ffill'].fillna(0) # fill NaN with 0
    resampled_data['ffill'] = resampled_data['ffill'] == 1 # only ffill if the whole minute is ffill
    #resampled_data['ffill'] = resampled_data['ffill'].astype(float)

    # # Filter out rows that are not exactly on the minute in the original data
    # filtered_data = data[data.index.second == 0].copy()

    # # use the resampled ffill column in the filtered data
    # filtered_data['ffill'] = resampled_data['ffill']

    
    return resampled_data



def interpolate_data(data, interpolation_col, limit=5, bool_columns=[]):
    # interpolate the data within 5 minutes; not data is on a 1 minute basis
    # if it doesn't fill a complete gap, keep the NaN values and don't interpolate
    data = data.copy()
    interpolated_data = data.copy()
    method = 'linear'
    interpolated_data[[interpolation_col]+bool_columns] = interpolated_data[[interpolation_col]+bool_columns].interpolate(
        method=method, 
        limit=5, 
        limit_area='inside', 
        limit_direction='forward',
        )
    # get the mask of the NaN values
    consecutive_nans = consecutive_nans_until_not(data[interpolation_col], consecutive_count=5)
    # set the values to NaN where the consecutive NaNs are more than 5
    interpolated_data.loc[consecutive_nans, interpolation_col] = np.nan
    if bool_columns:
        # set the bool columns to 0 if the interpolation column is NaN
        interpolated_data.loc[consecutive_nans, bool_columns] = 0.0
        #interpolated_data[bool_columns] = interpolated_data[bool_columns].fillna(0)
    return interpolated_data


# def continuous_time_series(data, interpolation_col, bool_columns, starttime, endtime):
#     data = data.copy()

#     starttime, endtime = pd.Timestamp(starttime), pd.Timestamp(endtime)

#     # # ensure time is index in case it is not
#     # data.set_index('time', inplace=True)

#     # handle bool columns
#     for bool_col in bool_columns:
#         # convert bool columns to int
#         #data[bool_col] = data[bool_col].fillna(0)
#         data[bool_col] = data[bool_col].astype(float)


#     # resample the data
#     data = resample_data(data)

#     # ensure that the data is from start to end - should not be necessary
#     data = ensure_data_is_from_start_to_end(data, starttime, endtime, bool_columns)

#     # interpolate the data within 5 minutes
#     data = interpolate_data(data, interpolation_col, limit=5) if interpolation_col else data

#     for bool_col in bool_columns:
#         # if larger than 0, set to True to indicate errors in the raw value
#         data[bool_col] = data[bool_col] > 0

#     return data


def process_with_adjusted(data, starttime, endtime, error_indicators, metadata):
    data = data.copy()

    source = metadata['Source']
    window_frozen = metadata['window_frozen']
    ### Adjusted processing
    # ensure time is datetime
    data['time'] = pd.to_datetime(data['time'])
    # set time to index
    data.set_index('time', inplace=True)

    # handle bool columns as floats
    for bool_col in error_indicators + ['ffill']:
        # convert bool columns to float
        data[bool_col] = data[bool_col].astype(float) # turn T/F to 1/0

    # resample the data
    data = resample_data(data)

    if source == "iFix":
        data = handle_ifix_data_saving(data, error_indicators, window_frozen)

    # ensure that the data is from start to end
    data = ensure_data_is_from_start_to_end(data, starttime, endtime)
    # now set nan to false for ffill, probably edge cases.
    #data['ffill'] = data['ffill'].fillna(False)

    # interpolate the data within 5 minutes
    # raw and error columns
    data = interpolate_data(data, 'raw_value', limit=5, bool_columns=error_indicators)

    # clean data
    data = interpolate_data(data, 'value_no_errors', limit=5, bool_columns=[])


    for bool_col in error_indicators:
        # if larger than 0, set to True to indicate errors in the raw value
        data[bool_col] = data[bool_col] > 0

    # error indicators should be False if the raw value is NaN
    data.loc[data['raw_value'].isna(), error_indicators] = False


    data.reset_index(drop=False, inplace=True, names='time')    

    return data



