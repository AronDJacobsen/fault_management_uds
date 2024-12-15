
import numpy as np
import pandas as pd
from fault_management_uds.data.format import merge_intervals


from fault_management_uds.config import REFERENCE_DIR



def add_rain_event_priority(data, dataset_args):
    # load rain events
    rain_events = pd.read_csv(REFERENCE_DIR / 'events' / 'rain_events.csv')
    rain_events['start'], rain_events['end'] = pd.to_datetime(rain_events['start']), pd.to_datetime(rain_events['end'])
    
    # adjust start and end times based on 
    # - sequence length
    # - average response time
    rain_events['start'] = rain_events['start'] - pd.Timedelta(minutes=dataset_args['sequence_length'])
    rain_events['end'] = rain_events['end'] + pd.Timedelta(minutes=60*3)
    rain_events = merge_intervals(rain_events)

    # Generate the complete date range
    complete_range = pd.concat([
        pd.Series(pd.date_range(row['start'], row['end']))
        for _, row in rain_events.iterrows()
    ])

    # inject priority into the data
    rain_event_priority = dataset_args.get('rain_event_priority', 1)
    data['priority_weight'] = 1.0 # default
    data.loc[data.index.isin(complete_range), 'priority_weight'] = rain_event_priority
    return data


def add_feature_engineering(data, dataset_args):
    if ('sin_time' in dataset_args['engineered_vars']) and ('cos_time' in dataset_args['engineered_vars']):
        sin_time, cos_time = cyclic_time_of_day(data)
        data['sin_time'], data['cos_time'] = sin_time, cos_time
    if ('sin_day' in dataset_args['engineered_vars']) and ('cos_day' in dataset_args['engineered_vars']):
        sin_day, cos_day = cyclic_day_of_week(data)
        data['sin_day'], data['cos_day'] = sin_day, cos_day
    return data

def cyclic_time_of_day(data):
    # add cyclical time of day features on a minute scale
    normalized_time = (data.index.hour + data.index.minute/60) / 24 # range [0, 1]
    sin_time = np.sin(2 * np.pi * normalized_time) + 1 # range [0, 2]
    cos_time = np.cos(2 * np.pi * normalized_time) + 1 # range [0, 2]
    return sin_time, cos_time

def cyclic_day_of_week(data):
    # add cyclical day of week features
    normalized_day = data.index.dayofweek / 7
    sin_day = np.sin(2 * np.pi * normalized_day) + 1 # range [0, 2]
    cos_day = np.cos(2 * np.pi * normalized_day) + 1 # range [0, 2]
    return sin_day, cos_day




