
import pandas as pd
from tqdm import tqdm




def classify_rain_events(rain_data, min_event_duration=30, min_event_rainfall=10, max_zero_gap=15):
    # Thresholds
    # min_event_duration = 30  # in minutes
    # min_event_rainfall = 10  # in mm
    # max_zero_gap = 15  # in minutes
    
    # Ensure time as the index (assuming your index is time-based)
    rain_data = rain_data.reset_index(drop=False, names='time')

    # Initialize variables for tracking events
    events = [] # List to store the events
    current_event = [] # List to store the current ongoing event
    zero_streak = 0 # Counter for the number of zero values
    
    for i, row in tqdm(rain_data.iterrows(), total=len(rain_data), desc='Classifying rain events'):
        # Check if it's raining
        if row['value'] > 0:
            # If there's no ongoing event, start a new one
            if not current_event:
                current_event_start_time = row.time
            current_event.append(row)
            zero_streak = 0  # Reset zero streak since it's raining
            
        # Check if it's not raining
        elif row['value'] == 0 and current_event:
            zero_streak += 1  # Increase zero streak
            if zero_streak <= max_zero_gap:
                # Still within the relaxed gap period, consider it part of the event
                current_event.append(row)
            else:
                # Zero streak exceeds allowed gap, event ends here, but take the before the zero streak
                event_end_time = rain_data.loc[i - zero_streak, 'time']
                
                # Create a DataFrame for the event and apply filtering
                event_df = pd.DataFrame(current_event)
                event_duration = round((event_df.time.max() - event_df.time.min()).total_seconds() / 60.0)  # in minutes
                total_rain = event_df['value'].sum()
                
                # Check if the event meets the criteria
                if event_duration >= min_event_duration and total_rain >= min_event_rainfall:
                    events.append({
                        'start': current_event_start_time,
                        'end': event_end_time,
                        'duration': event_duration,
                        'rainfall': total_rain
                    })
                
                # Reset the current event tracking
                current_event = []
                zero_streak = 0

    # After looping, check if there's a valid ongoing event
    if current_event:
        event_df = pd.DataFrame(current_event)
        event_duration = (event_df['time'].max() - event_df['time'].min()).total_seconds() / 60.0
        total_rain = event_df['value'].sum()
        if event_duration >= min_event_duration and total_rain >= min_event_rainfall:
            events.append({
                'start': current_event_start_time,
                'end': event_df['time'].max(),
                'duration': event_duration,
                'rainfall': total_rain
            })
    
    return pd.DataFrame(events)


