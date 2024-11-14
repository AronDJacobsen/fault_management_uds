import os
import pytz

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from datetime import datetime, timedelta
from scipy.signal import find_peaks


#from fault_management_uds.data.process import ensure_data_is_from_start_to_end



############################################################################################################
############################################################################################################
# Below are useful functions form the Bellinge project


"""
This module contains functions for importing data from different sources and handling the data.

Everything is provided by the Bellinge project.

@article{nedergaard2021a,
  author = {Nedergaard Pedersen, Agnethe and Wied Pedersen, Jonas and Vigueras-Rodriguez, Antonio and Brink-Kjær, Annette and Borup, Morten and Steen Mikkelsen, Peter},
  title = {The Bellinge data set: open data and models for community-wide urban drainage systems research},
  language = {eng},
  format = {article},
  journal = {Earth System Science Data},
  volume = {13},
  number = {10},
  pages = {4779-4798},
  year = {2021},
  issn = {18663508, 18663516},
  publisher = {Copernicus Publications},
  doi = {10.5194/essd-13-4779-2021}
}

"""


############################################################################################################
# Handle iFIX data saving
############################################################################################################


def ifix_forward_fill(data, value_columns, bool_columns, frozen_threshold, obvious_min, maximum_threshold):
    # Fix the iFix forward fill stuff..
    # - the forward fill is not always correct, so we need to adjust it
    # - instead of forward filling all values, only forward fill if 
    #     - it does not exceed the frozen threshold
    #     - it if is a relatively low value, becuase they don't fluctuate that much
    # additionally, forward fill error indicators, since they will be propagated as well

    data = data.copy()
    # #print(f"Number of forward fill values originally: {data['ffill'].sum()}")
    # # check that the forward fill does not exceed the frozen value threshold
    # # i.e. in case ffill is True for more than the frozen_threshold - it must instead be a frozen value
    # ffill_not_frozen = handle_ffill_frozen_threshold(data, time_column='time', bool_column='ffill', minute_threshold=frozen_threshold)
    # #print(f"Number of forward fill values that are not frozen: {ffill_not_frozen.sum()}")

    # # has to be lower than the low value threshold
    # # and obvious minimum has to be set to a value that is not too low (e.g. flow sensor has -5)
    # minimum_threshold = max(obvious_min, -maximum_threshold) # in case the obvious min is lower than the inverted max, use the inverted max
    # ffill_low = allow_ffill_low_values(data, value_column='raw_value', bool_column='ffill', minimum_threshold=minimum_threshold, maximum_threshold=maximum_threshold)
    # #print(f"Number of forward fill values that are low but ok: {ffill_low.sum()}")

    # # allow forward fill if it is either not frozen or not low
    # ffill = (ffill_not_frozen | ffill_low).tolist()
    # #print(f"Number of forward fill values that are not frozen or low: {sum(ffill)}")

    # # add True one column back for including it in the forward fill, and remove ffill if previous is Nan
    # ffill = add_true_one_back(ffill)
    ffill_mask = data['ffill']
    print(f"    Number of forward fill values originally: {ffill_mask.sum()}")
    # forward fill the data based on the ffill indicator
    data.loc[ffill_mask, value_columns + bool_columns] = data.loc[ffill_mask, value_columns + bool_columns].ffill()
    print(f"    Not NaN values {data['raw_value'].notna().sum()} out of {len(data)}")

    return data#[value_column]


def allow_ffill_low_values(df, value_column, bool_column, minimum_threshold, maximum_threshold):
    # Given a value column and forward fill the values

    # Extract values that are currently set to forward fill
    ffill_values = df.loc[df[bool_column], value_column]
    # Check if the values are not within the specified range (outside the min-max threshold)
    outside_range = ~ffill_values.between(minimum_threshold, maximum_threshold) # returns a boolean mask
    
    # Set the forward fill flag to False where values are outside the range
    df.loc[ffill_values.index[outside_range], bool_column] = False

    return df[bool_column]#.tolist()


def handle_ffill_frozen_threshold(df, time_column='time', bool_column='ffill', minute_threshold=5):
    # Goal: Switch consecutive True values to False if they span more than a certain time threshold
    df = df.copy().reset_index(drop=True)
    # Set the time threshold
    time_threshold = timedelta(minutes=int(minute_threshold))
    
    # Initialize variables
    start_time = None
    count = 0
    
    for i in range(len(df)):
        if df.loc[i, bool_column]:  # If the current value is True
            if count == 0:
                # Start new sequence of consecutive Trues
                start_time = df.loc[i, time_column]
            # Increment the count of consecutive Trues
            count += 1
        else:       
            # A switch to False breaks the sequence     
            if count > 0:
                # Check if the time difference exceeds the threshold
                time_diff = df.loc[i-1, time_column] - start_time
                if time_diff >= time_threshold:
                    # If the time difference meets or exceeds the threshold, switch to False
                    df.loc[i - count:i, bool_column] = False
            # Reset the count
            count = 0

    # Check if the last sequence of Trues exceeds the threshold
    if count > 0:
        time_diff = df.loc[len(df) - 1, time_column] - start_time
        if time_diff >= time_threshold:
            # If the time difference meets or exceeds the threshold, switch to False
            df.loc[len(df) - count:len(df), bool_column] = False
    
    return df[bool_column]#.tolist()



def add_true_one_back(series):
    # add a true value one back, but not if the previous value is NaN
    # - this makes sense, because if not then e.g. [...0.1...NaN...], would have [...True...True...]
    # - this function is only meant to adjust from value: [...0.1, Nan...], before ffill [...False, True...] to ffill: [...True, True...], so that the ffill can be applied
    for i in range(1, len(series)): # start from 1
        # if true, set the previous to true
        if series[i]:
            # TODO: If the previous is False and NaN, don't set it to True
            # we'll set this no matter what, even if start value is NaN
            series[i-1] = True
    return series




############################################################################################################
# Importing data
############################################################################################################


def load_raw_data(i, obs_input, data_path, original_path):
    # change to the correct directory
    os.chdir(str(data_path))

    # extract variables
    name = obs_input.at[i,'IdMeasurement']
    path = obs_input.at[i,'Folderpath'] + obs_input.at[i,'Filename']
    conversion = obs_input.at[i,'Conversion']

    if obs_input.at[i,'Source'] == 'System2000':
        raw_data = import_system2000(path, conversion)#, obs_input.at[i,'StartTime'], obs_input.at[i,'EndTime'] )
        print("    System2000 data is loaded:", obs_input.at[i,'IdMeasurement']) 
    elif obs_input.at[i,'Source'] == 'iFix':  
        raw_data = import_ifix(path, conversion)#, obs_input.at[i,'StartTime'], obs_input.at[i,'EndTime'] )
        print("    iFix data is loaded:", obs_input.at[i,'IdMeasurement'])
        
    elif obs_input.at[i,'Source'] == 'Danova':  
        raw_data = import_danova(path, conversion)#, obs_input.at[i,'StartTime'], obs_input.at[i,'EndTime'] )
        print("    Danova data is loaded:", obs_input.at[i,'IdMeasurement'])       
    else: 
        print("    There is no data")

    # NOTE: the rounding function has been removed here

    # change back to the correct directory
    os.chdir(str(original_path))

    return raw_data


#Read System2000 data
def import_system2000(path, conversion):
    #System2000
    df = pd.read_csv(path, sep=";",header=0, skipinitialspace=True)
    
    #Change time to UTC
    dst_boolean = df['BASE'] == (4096 or 6144) # if 0 it is winter time, if 4096 it is summer time  
    dst_boolean = dst_boolean.to_numpy()

    pd_time_array = pd.to_datetime(list(df['TIME']),format="%Y.%m.%d %H:%M:%S") # turns a string into a datetime object
    pd_time_array = pd_time_array.tz_localize(tz='Europe/Copenhagen',ambiguous=dst_boolean).tz_convert('UTC') # tz_localize sets the time zone of a time stamps, tz_convert changes a time to another zone
    pd_time_array = pd_time_array.tz_convert(None)
    
    raw_data = pd.DataFrame({'time': pd_time_array,
                         'value': df.iloc[:,2]})

    #Convert to same unit
    raw_data['value']=raw_data['value']/conversion #convert to m
    
    
    return raw_data

def import_ifix(path, conversion):
    #IFIX
  
    dfI = pd.read_fwf(path, decimal=",", sep='\t', parse_dates=True, skiprows=(1), skipfooter=(3))
    dfI.columns = ['time','value','Quality'] #Nyt navn til kolonner da gamle er mærkelige

    #Convert scientific number with commas with dots
    dfI['value'] = pd.to_numeric(dfI['value'].apply(lambda x: re.sub(',', '.', str(x))))
    dfI['time'] = pd.to_datetime(dfI['time'], format="%Y-%m-%d %H:%M:%S.%f") #Gør så data bliver i rigtig dato format

    #Change time to UTC - easiest for now!
    dfI['UTCtime'] =  FromDanishDSTtimeToUTC(dfI['time'])
    #dfI['UTCtime'] = from_danish_dst_to_utc(dfI['time'])['time']
    
            
    raw_data = pd.DataFrame({'time': dfI['UTCtime'],
                         'value': round(dfI['value'],3), #round data
                         'quality_ifix':dfI['Quality']})
    
    raw_data['value']=raw_data['value']/conversion #convert to m

    return raw_data


def import_danova(path, conversion):
    df = pd.read_csv(path, decimal=",", sep='\t', parse_dates=True, skiprows=1, 
                    encoding='latin1',
                     usecols=[0,1])
    df.columns = ["time", "value"]
    
    #Make the time in correct format
    df['time'] = pd.to_datetime(df['time'], format="%d-%m-%Y %H:%M:%S")
#    df = df.set_index('time')

    #Change time to UTC - easiest for now!
    df['UTCtime'] =  FromDanishDSTtimeToUTC(df['time'])
    #df['UTCtime'] = from_danish_dst_to_utc(df['time'])['time']
    df['value']=df['value']/conversion #convert to m
 
        
    raw_data = pd.DataFrame({'time': df['UTCtime'],
                         'value': df['value']})
    
    return raw_data



############################################################################################################
# Handle iFIX data
############################################################################################################



def handle_ifix_data_saving(data, error_indicators, window_frozen):

    original_ffill = data['ffill'].copy()
    data = data.copy()

    # Detect ffill groups
    data['groups'] = (data['ffill'] & ~data['ffill'].shift(fill_value=False)).cumsum()
    data['groups'] = data['groups'] * data['ffill'] # Set groups to 0 if ffill is False

    def interpolate_blocks(group, value_col, bool_columns=[]):
        # get time difference between start and end of the group
        time_diff = group.index[-1] - group.index[0]

        # if it ends with erroneous values, or exceeds the frozen window threshold; use ffill
        if (new_group[value_col].isna().iloc[-buffer:].all()) or (time_diff > pd.Timedelta(minutes=window_frozen)):
            group.loc[:, [value_col] + bool_columns] = group.loc[:, [value_col] + bool_columns].ffill()
        # if shorter than window_frozen, assume a dynamic environment, interpolate with time
        else:
            # interpolate based on time
            group.loc[:, [value_col] + bool_columns] = group.loc[:, [value_col] + bool_columns].interpolate(method='time', limit_area='inside')
        return group

    # interpolate the raw and error values within the ffill blocks
    #data[error_indicators] = data[error_indicators].astype(float)
    for group_idx, group in data.groupby('groups'):
        # only interpolate if the group is not the first one
        if group_idx == 0:
            continue

        # Get the start and end indices of the group using iloc
        start_idx = group.index[0]  # First index of the group
        end_idx = group.index[-1]    # Last index of the group
        
        # Get the positions of these indices
        start_pos = data.index.get_loc(start_idx)  # Integer position of start index
        end_pos = data.index.get_loc(end_idx)      # Integer position of end index

        # add a buffer to the start and end positions
        buffer = 2
        start_pos = max(0, start_pos - buffer)  # Ensure we don't go below 0
        end_pos = min(len(data) - 1, end_pos + buffer)  # Ensure we don't go above the last index

        # extract the group
        new_group = data.iloc[start_pos:end_pos + 1].copy()  # Include the end position
        
        # ensure that there is a valid value are within the buffer start and end positions
        # i.e. if a buffer of 2, then e.g. one valid value within the first and last two positions
        # if not new_group['raw_value'].isna().iloc[0:buffer].all() or new_group['raw_value'].isna().iloc[-buffer:].all():
        #     # interpolate the raw value and error indicators within the group
        new_group = interpolate_blocks(new_group, value_col='raw_value', bool_columns=error_indicators)
        # handle data saving only if some value within the buffer is not NaN
        if not new_group['value_no_errors'].isna().iloc[0:buffer].all():
            # interpolate the value_no_errors
            new_group = interpolate_blocks(new_group, value_col='value_no_errors', bool_columns=[])

        # insert the interpolated group back into the original data without the buffer
        data.loc[group.index, ['raw_value', 'value_no_errors'] + error_indicators] = new_group.loc[group.index, ['raw_value', 'value_no_errors'] + error_indicators]

    # Drop helper columns and restore original ffill column
    data = data.drop(columns=['groups'])

    return data


def add_missing_1min_timepoints(data, start, end):

    # round start and end to the nearest minute
    start, end = start.floor('min'), end.ceil('min')

    # Create the complete time range with 1-minute intervals
    idx = pd.date_range(start=start, end=end, freq='1min')

    # Find the timestamps in `idx` that are not already in `data.index`
    missing_idx = idx.difference(data.index)

    # Create a new index that combines `data.index` with the missing timestamps from `idx`
    new_index = data.index.union(missing_idx).sort_values()

    # Reindex `data` to include all timestamps in `new_index`
    data = data.reindex(new_index)

    return data

# def Fill_data_savings_points(df, time='time', timeperiod='1min', col='value'):  

    # start, end = df[time].iloc[0], df[time].iloc[-1]
    # # set time to index
    # df.set_index(time, inplace=True)
    # df = add_missing_1min_timepoints(df, start, end)
    # df.reset_index(inplace=True, drop=False, names="time")
    
    # # create the ffill column, set it to True there 'value' is NaN
    # df['ffill'] = df[col].isna()

    # return df



############################################################################################################
# Data Quality Control
############################################################################################################



def data_quality_control(raw_data, i, obs_input, man_remove):
    name = obs_input.at[i,'IdMeasurement']
    # #SETTING FLAGS FOR ERRORS
    if obs_input.at[i,'Source'] == 'iFix': 
        data = Fill_data_savings_points(raw_data) 
        stamp_id = Stamped_error(data.value, data.quality_ifix, 0) #0 is error in data
    else:
        data = NoFill_data_savings_points(raw_data)
        stamp_id = pd.Series(np.zeros(len(data)),dtype=bool)
    #print("Stamp_id and padding of datasaving values is calculated")
    
    man_remove_id = Remove_man(data, man_remove, name)
    
    outbound_id = Out_of_bounds(data.value, obs_input.at[i,'obvious_min'], obs_input.at[i,'obvious_max'])
    #print("Outbound_id is calculated")
    
    # array(['Level', 'Position', 'Discharge', 'Power'], dtype=object)
    # avoid sensors with name LevelPS, and those of type Position, Discharge, and Power
    avoid_sensor_names = ['G71F68Y_LevelPS']
    avoid_sensor_types = ['Position', 'Power']
    sensor_type = obs_input.at[i,'Type']
    if name in avoid_sensor_names or sensor_type in avoid_sensor_types:
    #if name == 'G71F68Y_LevelPS' or obs_input.at[i,'Type'] == 'Position' or obs_input.at[i,'Type'] == 'Flow':
        frozen_id2 = pd.Series(np.zeros(len(data.value)),dtype=bool)    

    else: 
        frozen_id2 = Frozen_sensor(data.value, data.time, 0, obs_input.at[i,'window_frozen'])
    #print("Frozen_id is calculated")
    
   
    if obs_input.at[i,'Source'] == 'Danova': 
        outlier_id = Outlier_point(data.value, obs_input.at[i,'outlier_threshold'], obs_input.at[i,'outlier_width'])
    else: 
        outlier_id = pd.Series(np.zeros(len(data.value)),dtype=bool)
    #print("Outlier_id is calculated")

    value_no_errors = data.value.copy()
    value_no_errors[stamp_id | outbound_id | frozen_id2 | man_remove_id | outlier_id] = np.nan  

    # make list with errors instead with numbers
    data_with_tags = pd.DataFrame({'time': data.loc[:,'time'],
                              'raw_value': data.loc[:,'value'],
                              'value_no_errors': value_no_errors, # here the errors are set to nan
                              'ffill': data.loc[:,'ffill'],
                              'man_remove': man_remove_id,
                              'stamp': stamp_id,
                              'outbound': outbound_id,
                              'frozen': frozen_id2,
                              'outlier': outlier_id
                              })

    return data_with_tags






def Fill_data_savings_points(df, time='time', timeperiod='1min', col='value'):  
    """Goal: Fill in missing time periods in the dataframe if the time period is larger than 1 minute, using forward fill."""
    
    # Ensure the time column is in datetime format
    df[time] = pd.to_datetime(df[time])
    # Sort the DataFrame by time
    df = df.sort_values(by=time).reset_index(drop=True)

    # Create a new DataFrame to store the missing periods
    new_rows = []
    
    # Iterate through the DataFrame and identify missing periods
    for i in range(len(df) - 1):
        time_diff = df[time][i+1] - df[time][i]
        
        # If the gap between consecutive time points is greater than 1 minute
        if time_diff > pd.Timedelta(timeperiod):

            start_time = df[time][i] + pd.Timedelta(timeperiod)
            # Generate the missing time periods between current and next time point
            missing_periods = pd.date_range(start=start_time, end=df[time][i+1], freq=timeperiod)
            # Append the missing periods with 'ffill' column as True
            for time_point in missing_periods:
                new_rows.append({time: time_point, 'ffill': True})
    
    # Convert the new rows into a DataFrame
    new_df = pd.DataFrame(new_rows)

    # Add 'ffill' column as False for original rows if not present
    if 'ffill' not in df.columns:
        df['ffill'] = False

    # If new_df is not empty, merge with the original dataframe
    if not new_df.empty:
        # Remove rows in new_df that have timestamps already present in df
        new_df = new_df[~new_df[time].isin(df[time])]
        # Concatenate original DataFrame with the new missing periods DataFrame
        df = pd.concat([df, new_df], axis=0).sort_values(by=time).reset_index(drop=True)
    
    return df





def NoFill_data_savings_points(df, time='time', Timeperiod = '1min'): #df is a dataframe, time is name of time column, Timeperiod is min time period
    flag_pad = pd.Series(np.zeros(len(df.value)),dtype=bool)    
    output = pd.DataFrame({'time': df['time'],
                         'value': df['value'],
                         'ffill': flag_pad})
    
    return output




#Check if the data is stamped errorous
#qualityerror is given value, which is an error-measurement
# Gives True value if value is None
def Stamped_error(values, quality, qualityerror):
    """Goal: Check if the data is stamped errorous by the ifix system."""
    flag_stamp = pd.Series(np.zeros(len(values)),dtype=bool)
    flag_stamp.loc[(quality == qualityerror)]= True
    
    return flag_stamp

#Flag out manually removed periods from table. 
def Remove_man(data, man_remove, name):
    """Goal: Remove manually removed periods from the data."""

    listremove = list(np.where(man_remove['IdMeasurement']==name))
    man_remove1 = man_remove.drop(man_remove[man_remove.IdMeasurement !=name].index).reset_index(inplace=False)
    
    flag_removeman = pd.Series(np.zeros(len(data.value)),dtype=bool)
    if np.asarray(listremove).size != 0:
        for n in range(0,len(man_remove1)):
            start= pd.to_datetime(man_remove1.at[n,'StartTime'], format = "%d-%m-%Y %H:%M")
            end = pd.to_datetime(man_remove1.at[n,'EndTime'], format = "%d-%m-%Y %H:%M")
           
            #Ensures that data is present for this file
            if sum(data['time']>=start) > 0:
                startid = data.loc[data['time']>=start].first_valid_index()
                if sum(data['time']>=end) >0:
                    endid = data.loc[data['time']>=end].first_valid_index()
                else: 
                    endid = len(data)
                flag_removeman.iloc[startid:endid] = True
            elif sum(data['time']>=end) >0:
                startid = 0
                endid = data.loc[data['time']>=end].first_valid_index()
                
                flag_removeman.iloc[startid:endid] = True
            else: 
                pass
                
    return flag_removeman


# Set physically realistic upper and lower values that the data is not allowed to exceed
# Any value outside the realistic range is flagged.
def Out_of_bounds(values,lower,upper):
    flag_outbound = pd.Series(np.zeros(len(values)),dtype=bool)
    flag_outbound.loc[values < lower] = True
    flag_outbound.loc[values > upper] = True
    return flag_outbound

# Check if the sensor has frozen and only delivers constant values.
# The user provides a time window that specifies the maximum number of neighboring data points that are allowed to be identical, before it seems suspicious.
# "window_size" of 2 means that the two points before and after a data point (5 in total) are checked if they are identical/frozen.
def Frozen_sensor(values, times, thres_value, window_size): 
    #values=data.value.copy()
    #times=data.time.copy()
    flag_frozen = pd.Series(np.zeros(len(values)),dtype=bool)
    values_diff = values.diff()
    values_diffminus = values.diff(-1)
    
    values_mean = round(abs(values_diff).rolling(window_size-1).mean(),4)
    values_diffmean = round(abs(values_diffminus).rolling(window_size-1).mean(),4)
    
    flag_frozen.loc[(values_mean == 0) & (values_diff == 0) & (values > thres_value)] = True
    flag_frozen.loc[(values_diffmean == 0) & (values_diffminus == 0) & (values > thres_value)] = True
    
    
    #ensures that the period is above window_size
    flag_frozen_diff = flag_frozen.astype(int).diff()
    s = flag_frozen_diff[flag_frozen_diff ==1].reset_index()
    e = flag_frozen_diff[flag_frozen_diff ==-1].reset_index()

    if len(s) != len(e):
        #e = e.append({'index':(len(flag_frozen_diff)-1)}, ignore_index=True)
        e = pd.concat([e, pd.DataFrame([{'index': (len(flag_frozen_diff)-1)}])], ignore_index=True)


    #timediff = pd.Series(np.zeros(len(s)))
    # Initialize with Timedelta dtype
    timediff = pd.Series([pd.Timedelta(0)] * len(s))

    # Loop to calculate time differences (if loop is required)
    for t in range(len(s)):
        timediff[t] = times[e['index'].iloc[t]] - times[s['index'].iloc[t]]


    flag_frozen2 = pd.Series(np.zeros(len(values)),dtype=bool)
    for t in range(0,len(s)):
        if timediff[t] >  pd.Timedelta(minutes=window_size):
            flag_frozen2[int(s['index'].iloc[t]):int(e['index'].iloc[t])] = True
   
    return flag_frozen2

# Check for outliers by looking if there is a sudden increase and drop in values
def Outlier_point(values, threshold1, width1):
    #Flag if there is a peak value of threshold (height) and width (0.2 and 1 respectively)
    flag_outlier = pd.Series(np.zeros(len(values)),dtype=bool)
    peaks, _ = find_peaks(values, threshold=threshold1, width=width1)
    flag_outlier[peaks] = True
  
    return flag_outlier





############################################################################################################
# Handle date and timezones
############################################################################################################



def from_danish_dst_to_utc(input_times):
    """
    Convert a list or DataFrame of datetime objects in Danish DST to UTC.
    
    Parameters:
    - input_times: A list of datetime objects or a pandas Series.
    
    Returns:
    - A pandas DataFrame with the UTC-converted times.
    """
    # Define Danish timezone with DST awareness
    danish_tz = pytz.timezone('Europe/Copenhagen')
    
    # Convert input times to UTC, handling DST automatically
    utc_times = [danish_tz.localize(t).astimezone(pytz.UTC) for t in input_times]
    
    # Return as DataFrame for compatibility with original function's output
    return pd.DataFrame(utc_times, columns=['time'])


def FromDanishDSTtimeToUTC(input_t):
    import datetime
    # Convert Flow data to UTC
    t=input_t
    # Create index values for shift in summer/witner time
    sshift9 = datetime.datetime(2009,3,29,2,00,00)
    sshift10 = datetime.datetime(2010,3,28,2,00,00)
    sshift11 = datetime.datetime(2011,3,27,2,00,00)
    sshift12 = datetime.datetime(2012,3,25,2,00,00)
    sshift13 = datetime.datetime(2013,3,31,2,00,00)
    sshift14 = datetime.datetime(2014,3,30,2,00,00)
    sshift15 = datetime.datetime(2015,3,29,2,00,00)
    sshift16 = datetime.datetime(2016,3,27,2,00,00)
    sshift17 = datetime.datetime(2017,3,26,2,00,00)
    sshift18 = datetime.datetime(2018,3,25,2,00,00)
    sshift19 = datetime.datetime(2019,3,31,2,00,00)
    sshift20 = datetime.datetime(2020,3,29,2,00,00)
    sshift21 = datetime.datetime(2021,3,28,2,00,00)
    wshift9 = datetime.datetime(2009,10,25,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift10 = datetime.datetime(2010,10,31,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift11 = datetime.datetime(2011,10,30,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift12 = datetime.datetime(2012,10,28,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift13 = datetime.datetime(2013,10,27,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift14 = datetime.datetime(2014,10,26,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift15 = datetime.datetime(2015,10,25,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift16 = datetime.datetime(2016,10,30,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift17 = datetime.datetime(2017,10,29,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift18 = datetime.datetime(2018,10,28,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift19 = datetime.datetime(2019,10,27,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift20 = datetime.datetime(2020,10,25,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshift21 = datetime.datetime(2021,10,31,2,00,00) # 2 o'clock instead of 3 o'clock as the wintertime change the time one hour back
    wshiftindex = [wshift9, wshift10, wshift11, wshift12, wshift13, wshift14, wshift15, wshift16, wshift17, wshift18, wshift19, wshift20, wshift21]
    shiftindex9 = np.where(np.logical_and(t>=sshift9,t<wshift9))[0]
    shiftindex10 = np.where(np.logical_and(t>=sshift10,t<wshift10))[0]
    shiftindex11 = np.where(np.logical_and(t>=sshift11,t<wshift11))[0]
    shiftindex12 = np.where(np.logical_and(t>=sshift12,t<wshift12))[0]
    shiftindex13 = np.where(np.logical_and(t>=sshift13,t<wshift13))[0]
    shiftindex14 = np.where(np.logical_and(t>=sshift14,t<wshift14))[0]
    shiftindex15 = np.where(np.logical_and(t>=sshift15,t<wshift15))[0]
    shiftindex16 = np.where(np.logical_and(t>=sshift16,t<wshift16))[0]
    shiftindex17 = np.where(np.logical_and(t>=sshift17,t<wshift17))[0]
    shiftindex18 = np.where(np.logical_and(t>=sshift18,t<wshift18))[0]
    shiftindex19 = np.where(np.logical_and(t>=sshift19,t<wshift19))[0]
    shiftindex20 = np.where(np.logical_and(t>=sshift20,t<wshift20))[0]
    shiftindex21 = np.where(np.logical_and(t>=sshift21,t<wshift21))[0]
    shiftindex = [*shiftindex9, *shiftindex10, *shiftindex11, *shiftindex12, *shiftindex13, *shiftindex14,*shiftindex15,
                  *shiftindex16, *shiftindex17, *shiftindex18, *shiftindex19, *shiftindex20, *shiftindex21]
    # Convert time series to UTC
    t_UTC = t - datetime.timedelta(hours=1) # Substract one hour from all data
    t_UTC1=pd.DataFrame(t_UTC, index=range(len(t_UTC)), columns=['time'])
    t_UTC1.iloc[shiftindex,0]=t_UTC1.iloc[shiftindex,0] - pd.Timedelta(hours=1) # Substract an extra hour at summer time

  
    # #Ensures that the double values in shift to wintertime is not located double. No matter what time a double value start as. 
    # for i in range(0, len(wshiftindex)): 
       
    #     #if int((wshiftindex[i]-pd.Timedelta(hours=1)) > t_UTC1.iloc[0]) & int(wshiftindex[i] < t_UTC1.iloc[-1]): #apply only for valid data
    #     #if (wshiftindex[i]-pd.Timedelta(hours=1) > t_UTC1.iloc[0]) and (wshiftindex[i] < t_UTC1.iloc[-1]):
    #     if (wshiftindex[i] - pd.Timedelta(hours=1) > t_UTC1.at[0, 'time']) and (wshiftindex[i] < t_UTC1.at[len(t_UTC1)-1, 'time']):

    #         tstart = t_UTC1.loc[t_UTC1['time']>(wshiftindex[i]-pd.Timedelta(hours=1))].iloc[[0,]].index[0]
    #         t_realend = t_UTC1.loc[t_UTC1['time']>(wshiftindex[i])].iloc[[0,]].index[0]
    #         tend = 0

    #         if t_realend > 0:
    #             for j in range(tstart, t_realend):
        
    #                 if t_UTC1.time[j] > t_UTC1.time[j+1]:
    #                     tend = j+1
    #                     t_UTC1.time[(tstart-1):tend]=(t_UTC1.time[(tstart-1):tend] - pd.Timedelta(hours=1))

    #                 else: 
    #                     pass
   
    # Ensures that the double values in shift to wintertime is not located double. No matter what time a double value starts as. 
    for i in range(0, len(wshiftindex)): 
        if (wshiftindex[i] - pd.Timedelta(hours=1) > t_UTC1.at[0, 'time']) and (wshiftindex[i] < t_UTC1.at[len(t_UTC1)-1, 'time']):
            
            tstart = t_UTC1.loc[t_UTC1['time'] > (wshiftindex[i] - pd.Timedelta(hours=1))].iloc[[0,]].index[0]
            t_realend = t_UTC1.loc[t_UTC1['time'] > wshiftindex[i]].iloc[[0,]].index[0]
            tend = 0

            if t_realend > 0:
                for j in range(tstart, t_realend):
                    if t_UTC1.time[j] > t_UTC1.time[j+1]:
                        tend = j + 1
                        # Update this line
                        t_UTC1.loc[(tstart-1):tend, 'time'] = t_UTC1.loc[(tstart-1):tend, 'time'] - pd.Timedelta(hours=1)
                    else:
                        pass
    return t_UTC1





############################################################################################################
############################################################################################################
# Old functions

def load_data(data_path, sensor):
    '''
    Load all data from a specific sensor, inlcuding all levels and SCADA systems

    Parameters:
    sensor (str): The name of the sensor

    Returns:
    dict_: A dictionary with all data from the sensor, arranged by level and SCADA system
    '''
    all_files = os.listdir(data_path)
    dict_ = {}
    for file in all_files:
        if sensor in file:
            if file.endswith('.Identifier'):
                continue
            # Set time columns as index
            df = pd.read_csv(data_path / file, index_col=0)
            info = file.split('_')
            level = 'NoLevel'
            for i in range(len(info)):
                if 'Level' in info[i]:
                    level = info[i]
                if 'System' in info[i]:
                    system = info[i]
                if 'Danovap' in info[i]:
                    system = info[i]
                if 'iFix' in info[i]:
                    system = info[i]
            dict_[sensor +'_'+ system+'_'+level] = df   
    return dict_


def load_timeseries_data(data_path, structure, sensor = None, startdate = '01-01-2009', enddate = '01-01-2024'):
    '''
    Load a timeseries from a specific sensor and level, concatenating different SCADA systems

    Parameters:
    structure (str): The structure name
    sensor (str): The sensor name of the sensor
    startdate (str): The start date of the timeseries
    enddate (str): The end date of the timeseries

    Returns:
    df: A dataframe timeseries
    '''
    dict_ = load_data(data_path, structure)
    if sensor == None:
        print('Specify level')
        return
    else:
        df = pd.DataFrame()
        for key in dict_.keys():
            if sensor in key:
                df = pd.concat([df, dict_[key]])
        df.time = pd.to_datetime(df.time)
        df = df.set_index('time')
        df = df.sort_index()
        df = df[startdate:enddate]
        idx = pd.date_range(start = df.index[0], end = df.index[-1], freq = '1min')
        df = df.reindex(idx, fill_value=np.nan)    
        return df


    
def filtering_dict(dict_, level = [None], part = [None]):
    '''
    Filters the ouput of load_data by the specified levels and SCADA systems (parts)

    Parameters:
    dict_ (dict): The output of load_data
    level (list): The levels to keep
    part (list): The SCADA systems to keep

    Returns:
    dict_filtered: A filtered dictionary
    '''
    dict_filtered = {}
    for key in dict_.keys():
        if level != [None]:
            split_ = key.split('_')
            level_ = split_[2]
            system = split_[1]
            for l in level:
                if l == level_:

                    if part != [None]:
                        for p in part:
                            if p in system:
                                dict_filtered[key] = dict_[key]
                    else:
                        dict_filtered[key] = dict_[key]
        elif part != None:
            for p in part:
                if p in key:
                    dict_filtered[key] = dict_[key]
        else:
            dict_filtered[key] = dict_[key]
    print('Filtered dict has {} keys'.format(len(dict_filtered.keys()))+ ':' + str(dict_filtered.keys()) + '\n')
    return dict_filtered




def custom_ffill(df, reference_col):
    '''
    Custom forward fill function that fills NaNs with the last value if the NaN period is less than 60 minutes

    Parameters:
    df (pd.DataFrame): The dataframe to fill
    reference_col (str): The column to fill

    Returns:
    result (pd.DataFrame): The filled dataframe
    '''
    # Make a copy of the dataframe to avoid modifying the original
    result = df.copy()
    
    # Identify periods of NaNs
    is_na = df[reference_col].isna()
    not_na_cumsum = (~is_na).cumsum()
    nan_periods = is_na.groupby(not_na_cumsum).cumsum()
    
    # Create a mask for periods of NaNs less than 60 minutes
    mask = (nan_periods < 60) & is_na
    
    # Apply forward fill where the mask is True
    result.loc[mask, reference_col] = result[reference_col].ffill()
    
    return result

def rain_window(df: pd.DataFrame, window_size: int = 60):
    '''
    Calculate the sum of the rain data in a window

    Parameters:
    df (pd.DataFrame): Should contain a column named 'rain' with the rain data
    window_size (int, optional): Defaults to 60.

    Returns:
    df (pd.DataFrame): The input dataframe with a new column 'rain_window' that contains the sum of the rain data in the window
    '''
    df['rain_window'] = df['rain'].rolling(window_size, min_periods = 1).sum()
    return df

def find_best_window(df: pd.DataFrame, start:int = 1, end:int = 300, jumps:int = 5, target:str = 'depth_s'):
    '''
    Find the window size that maximizes the correlation between the rain data and the target column
    - rolling (cumulative) back in time to find the best window size
    This will represent the depth better instead of instantaneous rain data.

    Parameters:
    df (pd.DataFrame): Should contain a column named 'rain' with the rain data
    start (int, optional): Defaults to 1. The minimum window size
    end (int, optional): Defaults to 300. The maximum window size
    jumps (int, optional): Defaults to 5. The amount of jumps between the start and end and speeds up the process
    target (str, optional): Defaults to 'depth_s'. The target column to correlate with the rain data

    Returns:
    best_window (int): The window size that maximizes the correlation
    best_corr (float): The maximum correlation found
    '''
    range_ = range(start, end, jumps)
    best_window = 0
    best_corr = 0
    for window in tqdm(range_):
        df['rain_window'] = df['rain'].rolling(window=window).sum()
        corr = df['rain_window'].corr(df[target])
        if abs(corr) > best_corr:
            best_corr = corr
            best_window = window
    return best_window, best_corr

def add_time_feature(df:pd.DataFrame):
    '''
    Add time features to the dataframe

    Parameters:
    df (pd.DataFrame): The dataframe to add the features to

    Returns:
    df (pd.DataFrame): The dataframe with the added features
    '''
    # Add sin and cosine of weekly, daily and monthly cycles
    df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['day_of_month_sin'] = np.sin(2 * np.pi * df.index.day / 30)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df.index.day / 30)
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    return df


def full_load_pipeline(df, important_col: str = 'depth_s', rain_start:int = 1, rain_end:int = 300, jumps:int = 5):
    '''
    Load a dataframe and apply all the necessary preprocessing steps
    - Forward fill NaNs in the important column
    - Find the best window size for the rain data
    - Calculate the rain window
    - Add time features

    Parameters:
    df (pd.DataFrame): The dataframe to preprocess
    important_col (str, optional): Defaults to 'depth_s'. The column to forward fill
    rain_start (int, optional): Defaults to 1. The minimum window size for the rain data
    rain_end (int, optional): Defaults to 300. The maximum window size for the rain data
    jumps (int, optional): Defaults to 5. The amount of jumps between the start and end and speeds up the process

    Returns:
    df (pd.DataFrame): The preprocessed dataframe
    '''
    
    rain = pd.read_csv('../data/Bellinge/rain_5.csv.gz', index_col = 0, parse_dates=True, compression='gzip')
    df['rain'] = rain # automatically aligns on the time index
    df = custom_ffill(df, important_col)
    df_train = df[:int(len(df)*0.8)]
    
    
    best, corr = find_best_window(df_train, start = rain_start, end = rain_end, jumps = jumps, target = important_col)
    df = rain_window(df, best)
    del rain
    
    df = add_time_feature(df)
    return df



    