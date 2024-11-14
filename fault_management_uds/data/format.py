
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

