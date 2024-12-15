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

import os
import tensorflow as tf


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




def visualize_indicator_dict(ax, individual_indicators, start=None, end=None, adjust='half-point', ylabel='Errors'):
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

    # if start and end are not none, set the x-axis limits and ticks
    if start is not None and end is not None:
        ax.set_xlim(start, end)
        ax = set_meaningful_xticks(ax, start, end)
    
    ax.set_yticks(np.arange(len(individual_indicators)))
    ax.set_yticklabels(individual_indicators.keys())
    ax.invert_yaxis()
    ax.set_title(ylabel)

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




def visualize_logs(logdir):
    # first check if the logdir exists
    if not logdir.exists():
        print(f"Logdir {logdir} does not exist")
        return
    event_files = [f for f in os.listdir(logdir) if f.startswith("events.out.tfevents")]
    assert len(event_files) == 1, "Multiple tensorboard files found"
    event_file = event_files[0]
    # Load the tensorboard logs
    logs = {
    }
    for e in tf.compat.v1.train.summary_iterator(str(logdir / event_file)):
        for v in e.summary.value:
            if v.tag in logs:
                logs[v.tag].append((e.step, v.simple_value))
            else:
                logs[v.tag] = [(e.step, v.simple_value)]
    print(f"Loggers: {logs.keys()}")
    tags = ['train_loss_step', 'train_loss_epoch', 'val_loss']

    # Plot the logs
    fig, ax = plt.subplots(figsize=(8, 4))
    lowest_first_value = np.inf
    min_step, max_step = np.inf, 0
    for tag in tags:
        values = logs[tag]
        steps, values = zip(*values)
        plt.plot(steps, values, label=tag, alpha=0.7, lw=1)
        lowest_first_value = min(lowest_first_value, values[0])
        min_step = min(min_step, steps[0])
        max_step = max(max_step, steps[-1])
    # update ylim
    ax.set_ylim(0, lowest_first_value * 2)
    ax.set_xlim(min_step, max_step)
    plt.legend()
    plt.tight_layout()
    # save
    plt.savefig(logdir / 'loss.png')
    plt.close()

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



