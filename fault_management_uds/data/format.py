
import numpy as np
import pandas as pd
from datetime import timedelta





def create_individual_indicators(indicator, indicator_2_meta, errors, bools_2_meta):
    """
    Create individual indicators from the indicator and error data

    Args:
        indicator (pd.Series): The indicator data.
        indicator_2_meta (dict): The mapping of indicator values to metadata.
        errors (pd.DataFrame): The error data.
        bools_2_meta (dict): The mapping of error values to metadata.
        
    Returns:
        dict: A dictionary of individual indicators with binary values and color mappings.
    """


    # Create indicator matrix
    # each row represents a data quality indicator

    individual_indicators = {}

    # iterate over each indicator
    if indicator is not None:
        for indicator_value, indicator_meta in indicator_2_meta.items():
            alias = indicator_meta['alias']
            individual_indicators[alias] = {}
            # create the binary indicator
            individual_indicators[alias]['indicator'] = (indicator == indicator_value).values.astype(int)
            # create color mapping
            individual_indicators[alias]['colormap'] = {1: indicator_meta['color'], 0: 'white'}
    
    
    
    if errors is not None:
        # iterate over each error
        for error, error_meta in bools_2_meta.items():
            alias = error_meta['alias']
            individual_indicators[alias] = {}
            # create the binary indicator
            individual_indicators[alias]['indicator'] = (errors[error] == True).values.astype(int)
            # create color mapping
            individual_indicators[alias]['colormap'] = {1: error_meta['color'], 0: 'white'}

    return individual_indicators


def create_indicator(sensor_data, value_col, error_data, error_indicators, start, end, obvious_min=0):
    """Sensor data has to be the clean data, and error data should be the bool data with ."""
    # 0 for no data as default
    sensor_data = sensor_data.copy()

    indicator = np.zeros(sensor_data.shape[0])

    # 1 for has data
    has_data_mask = sensor_data[value_col].notna().values
    indicator[has_data_mask] = 1

    # -1 for zero valued data
    # TODO: instead of a zero mask, use a 'minimum' mask! i.e. metadata, obvious_min
    zero_mask = sensor_data[value_col] <= obvious_min
    indicator[zero_mask] = -1

    # 2 for errors
    error_mask = error_data[error_indicators].any(axis=1).values
    indicator[error_mask] = 2

    # convert indicator into a df with time as index
    indicator = pd.DataFrame(indicator, index=sensor_data.index, columns=['value'])

    return indicator


# Function to merge overlapping intervals
def merge_intervals(df):

    # Sort the DataFrame by start time
    df = df.sort_values(by='start').reset_index(drop=True)

    if df.empty:
        return df

    merged_intervals = []
    current_start = df.loc[0, 'start']
    current_end = df.loc[0, 'end']
    current_rainfall = df.loc[0, 'rainfall']

    for i in range(1, len(df)):
        row_start = df.loc[i, 'start']
        row_end = df.loc[i, 'end']
        row_rainfall = df.loc[i, 'rainfall']

        if row_start <= current_end:  # Overlapping intervals
            current_end = max(current_end, row_end)
            # add the rainfall
            current_rainfall += row_rainfall
        else:
            # No overlap, add the current interval to the list
            merged_intervals.append({'start': current_start, 'end': current_end, 'rainfall': current_rainfall})
            # Start a new interval
            current_start = row_start
            current_end = row_end
            current_rainfall = row_rainfall

    # Append the last interval
    merged_intervals.append({'start': current_start, 'end': current_end, 'rainfall': current_rainfall})

    return pd.DataFrame(merged_intervals)



# # Function to merge overlapping intervals
# def merge_intervals(df, start_col='start', end_col='end', agg_columns=None):
#     """
#     Merge overlapping intervals in a DataFrame.

#     Parameters:
#     df (pd.DataFrame): Input DataFrame with interval data.
#     start_col (str): Name of the column representing the start of the interval.
#     end_col (str): Name of the column representing the end of the interval.
#     agg_columns (dict): Dictionary specifying aggregation functions for other columns. 
#                         Example: {'rainfall': 'sum', 'temperature': 'mean'}

#     Returns:
#     pd.DataFrame: DataFrame with merged intervals.
#     """
#     if agg_columns is None:
#         agg_columns = {}

#     # Sort the DataFrame by the start column
#     df = df.sort_values(by=start_col).reset_index(drop=True)

#     merged_intervals = []
#     current_interval = df.iloc[0].to_dict()

#     for i in range(1, len(df)):
#         row = df.iloc[i]

#         if row[start_col] <= current_interval[end_col]:  # Overlapping intervals
#             # Extend the current interval's end time
#             current_interval[end_col] = max(current_interval[end_col], row[end_col])
            
#             # Aggregate additional columns
#             for col, func in agg_columns.items():
#                 if func == 'sum':
#                     current_interval[col] += row[col]
#                 elif func == 'mean':  # Weighted mean for simplicity
#                     if 'count' not in current_interval:
#                         current_interval['count'] = 1  # Initialize count
#                     current_interval[col] = (current_interval[col] * current_interval['count'] + row[col]) / (current_interval['count'] + 1)
#                     current_interval['count'] += 1
#                 # Add more aggregation methods as needed
#         else:
#             # No overlap, add the current interval to the list
#             merged_intervals.append(current_interval)
#             current_interval = row.to_dict()

#     # Append the last interval
#     merged_intervals.append(current_interval)

#     # Drop helper columns if added (like 'count')
#     result_df = pd.DataFrame(merged_intervals)
#     if 'count' in result_df:
#         result_df.drop(columns=['count'], inplace=True)

#     return result_df


