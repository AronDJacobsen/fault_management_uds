from pathlib import Path

#import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from datetime import timedelta


from fault_management_uds.config import FIGURES_DIR, PROCESSED_DATA_DIR


def get_segment_start_end_color(row, indicator_2_color):
    """
    Get the start, end and color of each segment in the timeline.
    """
    segment_starts = np.flatnonzero(np.diff(row)) + 1  # segment start
    segment_starts = np.insert(segment_starts, 0, 0)
    segment_ends = np.append(segment_starts[1:], len(row)) 
    # get the color for each starting segment
    segment_color = np.vectorize(indicator_2_color.get)(row[segment_starts])
    return segment_starts, segment_ends, segment_color


def adjust_segments(segment_starts, segment_ends, adjust='half-point'):
    """Adjust segments to half-points based on integer positions."""
    # Adjust the segments to half-points
    if adjust == 'half-point':
        segment_starts = [s - 0.5 for s in segment_starts]
        segment_ends = [e - 0.5 for e in segment_ends]
    elif adjust == 'full-point':
        # shift start segments 1 minute back
        segment_starts = [s - 1 if s > 0 else s for s in segment_starts]
    # else if adjust i a integer, add that as a buffer
    elif isinstance(adjust, int):
        segment_starts = [s - adjust - 1 if s > 0 else s for s in segment_starts]
        segment_ends = [e + adjust for e in segment_ends]
    else:
        raise ValueError(f"Invalid adjust value: {adjust}")
    return segment_starts, segment_ends



def convert_to_matplotlib_dates(starts, ends, start):
    """Convert numeric offsets to datetime and adjust to half-points, then to matplotlib float format."""

    # if the number is a decimal, use timedelta on minutes and seconds
    starts_datetime = [start + timedelta(minutes=int(s), seconds=int((s % 1) * 60)) for s in starts]
    ends_datetime = [start + timedelta(minutes=int(e), seconds=int((e % 1) * 60)) for e in ends]

    starts_num = mdates.date2num(starts_datetime)
    ends_num = mdates.date2num(ends_datetime)
    return starts_num, ends_num


def visualize_error_timeline(ax, individual_indicators, start=None, end=None, adjust='half-point'):
    bar_width = 0.5
    for i, (indicator, indicator_meta) in enumerate(individual_indicators.items()):
        segment_starts, segment_ends, segment_color = get_segment_start_end_color(
            indicator_meta['indicator'], indicator_meta['colormap']
        )
        
        # Adjust the segments to half-points
        segment_starts, segment_ends = adjust_segments(segment_starts, segment_ends, adjust)
        
        if start is not None and end is not None:
            # Convert to matplotlib dates if start and end are provided
            segment_starts, segment_ends = convert_to_matplotlib_dates(segment_starts, segment_ends, start)

        # Convert to numpy array for broken_barh plot
        segments = np.column_stack((segment_starts, np.array(segment_ends) - np.array(segment_starts)))

        ax.broken_barh(segments, (i - bar_width / 2, bar_width), facecolors=segment_color)

    # # Set the x-limits if start and end are provided
    # if start is not None and end is not None:
    #     ax.set_xlim(mdates.date2num(start), mdates.date2num(end))
    
    ax.set_yticks(np.arange(len(individual_indicators)))
    ax.set_yticklabels(individual_indicators.keys())
    ax.invert_yaxis()
    ax.set_title('Errors')

    return ax





def visualize_error_span(ax, indicator, start, end, adjust='half-point', alpha=0.3):

    segment_starts, segment_ends, segment_color = get_segment_start_end_color(
        indicator['indicator'], indicator['colormap']
    )
        
    # Adjust the segments to half-points
    segment_starts, segment_ends = adjust_segments(segment_starts, segment_ends, adjust)
    
    if start is not None and end is not None:
        # Convert to matplotlib dates if start and end are provided
        segment_starts, segment_ends = convert_to_matplotlib_dates(segment_starts, segment_ends, start)
    
    # Use axvspan for each segment
    for s, e, c in zip(segment_starts, segment_ends, segment_color):
        #ax.axvspan(s, e, alpha=0.3, facecolor=c)
        ax.axvspan(s, e, alpha=alpha, facecolor=c)

    return ax




def set_meaningful_xticks(ax, start, end):

    # if the time span is less than a day (visualize on an hourly basis)
    if (end - start).days < 1:
        ax.xaxis.set_major_locator(mdates.DayLocator())  # Marks start and end with full date
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m-%Y'))  # Format in Day-Month-Year Hour:Minute
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=0, pad=5.5)#, labelcolor='navy')

        # Minor ticks: Hourly intervals
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Set hourly intervals
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))  # Format in Hour:Minute
        ax.tick_params(axis='x', which='minor', labelsize=10, rotation=0, pad=8)#, labelcolor='navy')

    # if the time span is less than a week (visualize on an 6-hour basis)
    elif (end - start).days < 7:
        ax.xaxis.set_major_locator(mdates.DayLocator())  # Marks start and end with full date
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d-%m-%Y'))  # Format in Day-Month-Year Hour:Minute
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=0, pad=5.5)#, labelcolor='navy')

        # Minor ticks: Hourly intervals
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # Set hourly intervals
        ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))  # Format in Hour:Minute
        ax.tick_params(axis='x', which='minor', labelsize=10, rotation=0, pad=8)#, labelcolor='navy')
    
    # if the time span is less than a month (visualize on a daily basis)
    elif (end - start).days < 30:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d\n%b\n%Y'))
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=0, pad=5.5)  
    
    # if the time span is less than a year (visualize on a monthly basis)
    elif (end - start).days < 365:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=0, pad=5.5)#, labelcolor='navy')

    # else, visualize on a yearly basis
    else:
        # Only visualize years using major ticks
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', which='major', labelsize=10, rotation=0, pad=5.5)
    return ax

def set_meaningful_xaxis_timestamps(ax, timestamps_df):

    # x-axis
    ax.set_xlim(0, len(timestamps_df))
    # Get indices of rows where 'time' is on January 1st at 00:00:00
    start_idx = timestamps_df[
        (timestamps_df['time'].dt.month == 1) &
        (timestamps_df['time'].dt.day == 1) &
        (timestamps_df['time'].dt.hour == 0) &
        (timestamps_df['time'].dt.minute == 0) &
        (timestamps_df['time'].dt.second == 0)
    ].index
    start_vals = timestamps_df.loc[start_idx, 'time'].dt.year.values
    # check if xticks will only be one year
    if len(start_idx) < 2:
        # plot on a monthly basis
        start_idx = timestamps_df[
            (timestamps_df['time'].dt.day == 1) &
            (timestamps_df['time'].dt.hour == 0) &
            (timestamps_df['time'].dt.minute == 0) &
            (timestamps_df['time'].dt.second == 0)
        ].index
        start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%m-%Y').values
        # check if xticks will only be one month
        if len(start_idx) < 2:
            # plot on a daily basis
            start_idx = timestamps_df[
                (timestamps_df['time'].dt.hour == 0) &
                (timestamps_df['time'].dt.minute == 0) &
                (timestamps_df['time'].dt.second == 0)
            ].index
            start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y').values
            # check if daily xticks will be less than 3 days
            if len(start_idx) < 3:
                # plot on every 12 hours
                start_idx = timestamps_df[
                    (timestamps_df['time'].dt.hour % 12 == 0) &
                    (timestamps_df['time'].dt.minute == 0) &
                    (timestamps_df['time'].dt.second == 0)
                ].index
                start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y %H:%M').values

                # check if every 12 hours xticks will be less than 2
                if len(start_idx) < 2:
                    # plot on every 6 hours
                    start_idx = timestamps_df[
                        (timestamps_df['time'].dt.hour % 6 == 0) &
                        (timestamps_df['time'].dt.minute == 0) &
                        (timestamps_df['time'].dt.second == 0)
                    ].index
                    start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y %H:%M').values
                    # check if every 6 hours xticks will be less than 2
                    if len(start_idx) < 2:
                        # plot on every 1 hour
                        start_idx = timestamps_df[
                            (timestamps_df['time'].dt.minute == 0) &
                            (timestamps_df['time'].dt.second == 0)
                        ].index
                        start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y %H:%M').values
                        # check if every 1 hour xticks will be less than 2
                        if len(start_idx) < 3:
                            # plot on every 30 minutes
                            start_idx = timestamps_df[
                                (timestamps_df['time'].dt.minute % 30 == 0) &
                                (timestamps_df['time'].dt.second == 0)
                            ].index
                            start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y %H:%M').values
                            # check if every 30 minutes xticks will be less than 2
                            if len(start_idx) < 2:
                                # plot on every 15 minutes
                                start_idx = timestamps_df[
                                    (timestamps_df['time'].dt.minute % 15 == 0) &
                                    (timestamps_df['time'].dt.second == 0)
                                ].index
                                start_vals = timestamps_df.loc[start_idx, 'time'].dt.strftime('%d-%m-%Y %H:%M').values


    # Set the x-ticks and labels
    ax.set_xticks(start_idx)
    ax.set_xticklabels(start_vals)
    return ax
    

def visualize_single_timeline(I, timestamps_df, variable, indicator_2_color, name_2_color, figsize=(16, 4)):
    """
    Visualize the timeline of the data with different indicators
    """

    # if I is 1D, convert it to 2D
    if len(I.shape) == 1:
        I = I.reshape(1, -1)

    # Transpose the 2D array I to have the variables as rows
    #I = I.copy().T

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=400)

    bar_width = 0.4


    # Iterate over each row in the 2D array I to plot segments
    for i, row in enumerate(I):
        segment_starts, segment_ends, segment_color = get_segment_start_end_color(row, indicator_2_color)
        segments = np.column_stack((segment_starts, segment_ends - segment_starts))
        # plot the segments
        ax.broken_barh(segments, (i - bar_width/2, bar_width), facecolors=segment_color)


    # Set the legend
    legend_elements = [Patch(facecolor=color, label=name) for name, color in name_2_color.items()]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    # expand the y-axis to make the plot look better
    ax.set_ylabel(variable, rotation=90, ha="center")
    ax.set_yticks([])  # Remove y-axis ticks and labels
    ax.set_ylim(-bar_width/2, bar_width/2)
    ax.invert_yaxis()


    # Set the x-axis
    ax = set_meaningful_xaxis_timestamps(ax, timestamps_df)

    # # add vertical lines for the years
    # for year in year_start_idx:
    #     ax.axvline(x=year, color='lightgray', linestyle='--', linewidth=0.5)
    # remove outer frame
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    # Show the plot
    plt.tight_layout()
    plt.show()



# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
