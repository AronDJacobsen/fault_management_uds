from pathlib import Path
import os
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from datetime import timedelta
import seaborn as sns
import itertools
from matplotlib.colors import Normalize

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from fault_management_uds.synthetic.synthetic_generator import find_unterrupted_sequences
from fault_management_uds.utilities import get_accelerator
from fault_management_uds.config import data_label_hue_map, data_label_hue_order, anomaly_hue_map, anomaly_hue_order

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
    tags = ['train_loss_epoch', 'val_loss']

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




### PCA


def fit_pca(outputs, all_feature_indices, feature_idx_names, data_label, max_components=20):
    # Fit
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(outputs[:, all_feature_indices])
    # Apply PCA
    max_components = 20
    n_components = min(max_components, len(all_feature_indices))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_scaled)
    # DataFrame with the PCA components
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    # add the coloring column
    pca_df['Data label'] = data_label 
    # Get the explained variance
    explained_variance = pca.explained_variance_ratio_
    # Get the loadings (components)
    loadings = pd.DataFrame(
        pca.components_.T,  # Transpose to match feature-column structure
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],  
        index=feature_idx_names
    )

    return pca_df, explained_variance, loadings


def loading_plot(coeff, labels, scale=1, colors=None, visible=None, ax=plt, arrow_size=0.5):
    # Plot the loadings
    for i, label in enumerate(labels):
        if visible is None or visible[i]:
            ax.arrow(
                0,
                0,
                coeff[i, 0] * scale,
                coeff[i, 1] * scale,
                head_width=arrow_size * scale,
                head_length=arrow_size * scale,
                color="#000" if colors is None else colors[i],
                )
            ax.text(
                coeff[i, 0] * 1.4 * scale,
                coeff[i, 1] * 1.4 * scale,
                label,
                color="#000" if colors is None else colors[i],
                ha="center",
                va="center",
                )


def pca_plot(pca_df, x, y, explained_variance, hue_col, hue_map, plot_loadings=False, loadings=None, save_folder=None):

    # Desired hue order
    hue_order = hue_map.keys()
    hue_order = list(hue_order)[::-1]

    # Example figure setup
    plt.figure(figsize=(9, 4))
    # Sort the data using np.vectorize to enforce order
    g = sns.scatterplot(
        data=pca_df.sort_values(hue_col, key=np.vectorize(hue_order.index)),  # Sort by hue_order index
        x=x, y=y, hue=hue_col, 
        hue_order=hue_order,  # Explicitly set hue_order
        palette=hue_map, alpha=0.6, s=5
    )


    # Add loadings
    if plot_loadings:
        loading_plot(loadings[[x, y]].values, loadings.index, scale=2, arrow_size=0.08)

    # Add variance explained by the
    g.set_xlabel(f"{x} ({explained_variance[0]*100:.2f}%)")
    g.set_ylabel(f"{y} ({explained_variance[1]*100:.2f}%)")

    # Reverse the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(hue_map),
               markerscale=2
    )
    if save_folder == None:
        plt.show()
    else:
        plt.savefig(save_folder / f'pca_{x}_{y}.png')
        plt.close()



def visualize_pca(save_folder, pca_df, explained_variance, loadings, max_components=4):

    # Variance explained
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', color='navy')
    plt.axhline(y=0.95, color='lightsteelblue', linestyle='--', linewidth=0.7)
    plt.text(0.99, 0.91, '95% variance explained', color = 'lightsteelblue', fontsize=10)
    idx_95 = np.argmax(np.cumsum(explained_variance) > 0.95)
    plt.axvline(x=idx_95 + 1, color='lightsteelblue', linestyle='--', linewidth=0.7)
    plt.xlabel('# Principal Components', fontsize=12)
    plt.xticks(np.arange(1, len(explained_variance) + 1))
    #plt.show()
    plt.savefig(save_folder / 'pca_expl_var.png')
    plt.close()



    # Plot the PCA components
    plot_pcs = list(pca_df.columns[list(range(max_components))])
    plot_pcs = ['PC1', 'PC2', 'PC3', 'PC4']
    pcs_combs = list(itertools.combinations(plot_pcs, 2))

    for x, y in pcs_combs:
        pca_plot(pca_df, x, y, explained_variance, 'Data label', data_label_hue_map,
                plot_loadings=False, loadings=None,
                save_folder=save_folder)


    # Plot the loadings
    n_components = idx_95+1 # seem to explain most of the variance
    plt.figure(figsize=(12, 5))
    sns.heatmap(loadings.iloc[:, :n_components].T, annot=False, cmap='coolwarm', center=0, fmt=".2f", cbar=True)
    plt.xlabel('Hidden dimension', fontsize=14)
    # rotate the y labels
    plt.yticks(rotation=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_folder / 'pca_loadings.png')
    plt.close()



# t-SNE
def fit_tsne(outputs, all_feature_indices, feature_idx_names, data_label, predicted_anomalies):

    # normalize the data
    # scaler = StandardScaler()
    # df_scaled = scaler.fit_transform(outputs[:, all_feature_indices])
    # if get_accelerator() == 'cuda':
    #     from tsnecuda import TSNE
    #     tsne = TSNE(n_components=2, random_seed=42)
    #     tsne_results = tsne.fit(df_scaled)
    #     tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])

    # # if CPU, apply PCA first
    # else:
    from sklearn.manifold import TSNE
    pca_df, explained_variance, _ = fit_pca(outputs, all_feature_indices, feature_idx_names, data_label)
    pca_df['Predicted'] = ['Anomalous' if x == 1 else 'Normal' for x in predicted_anomalies]
    
    # Subsample the data, stratified by the data label
    # - roughly 6% pollution, so the weights are gonna be 0.1 on the normal and 0.9 on the anomalies
    pca_df['Weights'] = np.where(pca_df['Data label'] != 'Original', 0.8, 0.2) # 0.2 for normal, 0.5 for anomalies
    sample_size = min(20000, len(pca_df))
    pca_df = pca_df.sample(n=sample_size, weights='Weights', random_state=42)
    
    # Store the data label and predicted
    data_label = pca_df['Data label'].values
    predicted_anomalies = pca_df['Predicted'].values
    # Remove columns
    pca_df = pca_df.drop(columns=['Data label', 'Predicted', 'Weights'])
    # Select n_components that explain 99% of the variance
    n_components = np.argmax(np.cumsum(explained_variance) > 0.95) + 1
    pca_df = pca_df.iloc[:, :n_components]     
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(pca_df)
    tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])

    tsne_df['Data label'] = data_label
    tsne_df['Predicted'] = predicted_anomalies

    return tsne_df

def visualize_tsne(save_folder, tsne_df):
    # Construct the t-SNE
    # Visualize with the data label
    plt.figure(figsize=(12, 7)) # it is based on 10 is the
    sns.scatterplot(x='t-SNE 1', y='t-SNE 2',
                    data=tsne_df.sort_values('Data label', key=np.vectorize(data_label_hue_order.index)),
                    hue='Data label', 
                    hue_order=list(data_label_hue_map.keys()),
                    palette=data_label_hue_map, s=6, alpha=0.7)
    plt.legend(loc='upper right')
    plt.savefig(save_folder / 't-SNE_data_label.png', dpi=150)
    plt.close()

    # Visualize with the predicted
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='t-SNE 1', y='t-SNE 2', 
                    data=tsne_df.sort_values('Predicted', key=np.vectorize(anomaly_hue_order.index)),
                    hue='Predicted', 
                    hue_order=list(anomaly_hue_map.keys()),
                    palette=anomaly_hue_map, s=6, alpha=0.7)
    plt.legend(loc='upper right')
    plt.savefig(save_folder / 't-SNE_predicted.png', dpi=150)
    plt.close()




### ROC, AUC and Confusion Matrix

def visualize_confusion(ax, i, key, conf_matrix, fmt, cmap):
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap, cbar=False, ax=ax)
    ax.set_title(key, fontsize=14)
    if i == 0:
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
    # set y and x ticks
    ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=10)
    ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=10)
    return ax


def visualize_roc_auc(ax, i, key, fpr, tpr, roc_auc):
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC: %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    if i == 0:
        ax.set_ylabel('TPR', fontsize=12)
        ax.set_xlabel('FPR', fontsize=12)
    ax.legend(loc='lower right')
    ax.set_title(key, fontsize=14)
    return ax




def cm_roc_auc_results(save_folder, results, evaluate_keys, data_label):
    # Overall results
    fig, axes = plt.subplots(3, len(evaluate_keys), figsize=(15, 7))
    auc_scores = {}
    for i, key in enumerate(evaluate_keys):
        # Confusion matrix
        conf_matrix = confusion_matrix(results['Actual'], results[key]['Predicted'])
        axes[0, i] = visualize_confusion(axes[0, i], i, key, conf_matrix, 'd', 'Blues')
        # Percentage confusion matrix
        conf_matrix = confusion_matrix(results['Actual'], results[key]['Predicted'])
        conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
        axes[1, i] = visualize_confusion(axes[1, i], i, ' ', conf_matrix, ".2f", 'Blues')
        # ROC curve
        fpr, tpr, _ = roc_curve(results['Actual'], results[key]['Decision Function'])
        roc_auc = auc(fpr, tpr)
        auc_scores[key] = {'Overall': roc_auc}
        axes[2, i] = visualize_roc_auc(axes[2, i], i, ' ', fpr, tpr, roc_auc)
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_folder / 'cm_roc_auc_Overall.png')
    plt.close()


    # For each anomaly
    for label in data_label_hue_order[::-1]:
        if label == 'Original':
            continue
        mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(2, len(evaluate_keys), figsize=(15, 5))
        for i, key in enumerate(evaluate_keys):
            # Confusion matrix
            conf_matrix = confusion_matrix(results['Actual'][mask], results[key]['Predicted'][mask])
            conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
            axes[0, i] = visualize_confusion(axes[0, i], i, key, conf_matrix, ".2f", 'Blues')

            # ROC curve
            fpr, tpr, _ = roc_curve(results['Actual'][mask], results[key]['Decision Function'][mask])
            roc_auc = auc(fpr, tpr)
            auc_scores[key][label] = roc_auc
            axes[1, i] = visualize_roc_auc(axes[1, i], i, ' ', fpr, tpr, roc_auc)

        plt.tight_layout()
        #plt.show()
        plt.savefig(save_folder / f'cm_roc_auc_{label}.png')
        plt.close()

        print('\n')

    return auc_scores



### The matrices

# def annotate_heatmap(data, data_fmt, ax, cmap='Blues', high_best=True):
#     norm = Normalize(vmin=data.min(), vmax=data.max())
#     cmap = plt.get_cmap(cmap)
    
#     for i in range(data.shape[0]):  # Iterate over rows
#         for j in range(data.shape[1]):  # Iterate over columns
#             value = data[i, j]
#             is_max = value == data[i].max() if high_best else value == data[i].min()
#             text_kwargs = {"weight": "bold"} if is_max else {}
            
#             # Get the background color for the cell
#             bg_color = cmap(norm(value))
            
#             # Determine text color (white or black) based on brightness
#             brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
#             text_color = "white" if brightness < 0.80 else "black"
            
#             # Add text annotation
#             str_value = data_fmt[i, j]
#             ax.text(j + 0.5, i + 0.5, str_value, ha="center", va="center", color=text_color, **text_kwargs)



# def visualize_metric_matrix(metric, data, cmap, round_to, suffix=None, high_best=True, figsize=(10, 3), save_folder=None):

#     # Prepare the data
#     data_fmt = data.to_numpy().round(round_to).astype(str)
#     if suffix is not None:
#         # add a % sign to each value
#         data_fmt = np.char.add(data_fmt, suffix)

#     # Create the heatmap
#     plt.figure(figsize=figsize)
#     ax = sns.heatmap(data, annot=False, cmap=cmap, cbar=False)
#     annotate_heatmap(data.to_numpy(), data_fmt, ax=ax, cmap=cmap, high_best=high_best)
#     # format
#     ax.yaxis.get_major_ticks()[-1].label1.set_fontweight('bold')
#     ax.yaxis.get_major_ticks()[-2].label1.set_fontweight('bold')
#     plt.gca().invert_yaxis()
#     plt.gca().xaxis.set_ticks_position('top')
#     plt.xticks(fontsize=12)
#     plt.gca().xaxis.set_tick_params(size=0)
#     plt.tight_layout()
#     if save_folder == None:
#         plt.show()
#     else:
#         plt.savefig(save_folder / f'metric_{metric}.png')
#         plt.close()


# def annotate_heatmap(data, data_fmt, ax, cmap='Blues', high_best=True, annotate_row_wise=True):
#     norm = Normalize(vmin=data.min(), vmax=data.max())
#     cmap = plt.get_cmap(cmap)
    
#     for i in range(data.shape[0]):  # Iterate over rows
#         for j in range(data.shape[1]):  # Iterate over columns
#             value = data[i, j]
#             is_max = value == data[i].max() if high_best else value == data[i].min()
#             text_kwargs = {"weight": "bold"} if is_max else {}
            
#             # Get the background color for the cell
#             bg_color = cmap(norm(value))
            
#             # Determine text color (white or black) based on brightness
#             brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
#             text_color = "white" if brightness < 0.80 else "black"
            
#             # Add text annotation
#             str_value = data_fmt[i, j]
#             ax.text(j + 0.5, i + 0.5, str_value, ha="center", va="center", color=text_color, **text_kwargs)

def annotate_heatmap(data, data_fmt, ax, cmap='Blues', high_best=True, annotate_row_wise=True):

    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.get_cmap(cmap)
    
    for i in range(data.shape[0]):  # Iterate over rows
        for j in range(data.shape[1]):  # Iterate over columns
            value = data[i, j]
            if annotate_row_wise:
                _high_best = high_best[i] if isinstance(high_best, list) else high_best
                is_max = value == data[i].max() if _high_best else value == data[i].min()
            else:  # Annotate column-wise
                _high_best = high_best[j] if isinstance(high_best, list) else high_best
                is_max = value == data[:, j].max() if _high_best else value == data[:, j].min()

            text_kwargs = {"weight": "bold"} if is_max else {}
            
            # Get the background color for the cell
            bg_color = cmap(norm(value))
            
            # Determine text color (white or black) based on brightness
            brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            text_color = "white" if brightness < 0.80 else "black"
            
            # Add text annotation
            str_value = data_fmt[i, j]
            ax.text(j + 0.5, i + 0.5, str_value, ha="center", va="center", color=text_color, **text_kwargs)


def visualize_metric_matrix(metric, data, cmap, round_to, suffix=None, high_best=True, figsize=(10, 3), save_folder=None, top_n_bold=2, annotate_row_wise=True, ysize=14):

    # round data
    data = data.round(round_to)

    # Prepare the data
    data_fmt = data.to_numpy().round(round_to).astype(str)
    if suffix is not None:
        # add a % sign to each value
        data_fmt = np.char.add(data_fmt, suffix)

    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data, annot=False, cmap=cmap, cbar=False)
    annotate_heatmap(data.to_numpy(), data_fmt, ax=ax, cmap=cmap, high_best=high_best, annotate_row_wise=annotate_row_wise)
    # format, top_n_bold is the number of top rows to be bold (y labels)
    for i in range(top_n_bold):
        ax.yaxis.get_major_ticks()[-(i+1)].label1.set_fontweight('bold')
    #ax.yaxis.get_major_ticks()[-2].label1.set_fontweight('bold')
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_ticks_position('top')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=ysize)
    # ensure y tick are horizontal
    plt.yticks(rotation=0)
    plt.gca().xaxis.set_tick_params(size=0)
    plt.tight_layout()
    if save_folder == None:
        plt.show()
    else:
        plt.savefig(save_folder / f'metric_{metric}.png', dpi=150)
        plt.close()




def get_coverage(predicted, anomaly_start_end):
    # Given start and end of anomalies, find how much of the anomaly is covered by the prediction
    # Find all indices where the value is 1
    ones_indices = np.where(predicted == 1)[0]
    # Find the closest index in both directions
    if len(ones_indices) == 0:
        # do not continue
        raise ValueError("No anomalies detected")

    coverage = []

    for start, end in anomaly_start_end:
        # extract relevant data
        anomaly_data = predicted[start:end+1]
        
        # Coverage
        coverage.append(anomaly_data.sum() / len(anomaly_data))

    return coverage

def get_timing(predicted, anomaly_start_end):
    # Given start and end of anomalies, find how close the prediction is to the start
    # Find all indices where the value is 1
    ones_indices = np.where(predicted == 1)[0]
    # Find the closest index in both directions
    if len(ones_indices) == 0:
        # do not continue
        raise ValueError("No anomalies detected")

    timing = []

    for i, (start, end) in enumerate(anomaly_start_end):
        # Filter indices that are after the start
        valid_indices = ones_indices[ones_indices > start]

        if valid_indices.size > 0:  # Check if there are valid indices
            distances = np.abs(valid_indices - start)
            closest_index = valid_indices[np.argmin(distances)]
            timing.append(closest_index - start)
        else:
            # stop the loop
            print(f"Stopping with {len(anomaly_start_end) - i} anomalies left")
            break
    return timing


def metric_results(save_folder, auc_scores, results, evaluate_keys, data_label):
    # row ordering
    row_order = data_label_hue_order[1:] + ['Average', 'Overall']

    # AUC
    auc_df = pd.DataFrame(auc_scores)
    # Calculate the average AUC as well
    auc_df.loc['Average'] = auc_df.loc[data_label_hue_order[1:]].mean()
    auc_df = auc_df.loc[row_order]
    visualize_metric_matrix('AUC', auc_df, 'Oranges', 2, suffix=None, figsize=(10, 3), save_folder=save_folder)

    # Precision
    precision_scores = {}
    for key in evaluate_keys:
        precision_scores[key] = {'Overall': precision_score(results['Actual'], results[key]['Predicted'], zero_division=0)*100}
        for label in data_label_hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            precision_scores[key][label] = precision_score(
                results['Actual'][mask], results[key]['Predicted'][mask], zero_division=0
            ) * 100
    precision_df = pd.DataFrame(precision_scores)
    precision_df.loc['Average'] = precision_df.loc[data_label_hue_order[1:]].mean()
    precision_df = precision_df.loc[row_order]
    visualize_metric_matrix('Precision', precision_df, 'Blues', 2, suffix='%', figsize=(10, 3), save_folder=save_folder)

    # Recall
    recall_scores = {}
    for key in evaluate_keys:
        recall_scores[key] = {'Overall': recall_score(results['Actual'], results[key]['Predicted'], zero_division=0)*100}
        for label in data_label_hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            recall_scores[key][label] = recall_score(
                results['Actual'][mask], results[key]['Predicted'][mask], zero_division=0
            ) * 100
    recall_df = pd.DataFrame(recall_scores)
    recall_df.loc['Average'] = recall_df.loc[data_label_hue_order[1:]].mean()
    recall_df = recall_df.loc[row_order]
    visualize_metric_matrix('Recall', recall_df, 'Reds', 2, suffix='%', figsize=(10, 3), save_folder=save_folder)

    # The other metrics
    # Get the start and end of each anomaly
    indices_of_ones = [index for index, value in enumerate(results['Actual']) if value == 1]
    _, anomaly_start_end = find_unterrupted_sequences(indices_of_ones, 0)

    # Coverage
    coverage_scores = {}
    for key in evaluate_keys:
        coverage_scores[key] = {'Overall': np.mean(get_coverage(results[key]['Predicted'], anomaly_start_end)) * 100}
        for label in data_label_hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            _indices_of_ones = [index for index, value in enumerate(results['Actual'][mask]) if value == 1]
            _, _anomaly_start_end = find_unterrupted_sequences(_indices_of_ones, 0)
            coverage_scores[key][label] = np.mean(get_coverage(results[key]['Predicted'][mask], _anomaly_start_end)) * 100
    coverage_df = pd.DataFrame(coverage_scores)
    coverage_df.loc['Average'] = coverage_df.loc[data_label_hue_order[1:]].mean()
    coverage_df = coverage_df.loc[row_order]
    visualize_metric_matrix('Coverage', coverage_df, 'Purples', 2, suffix='%', figsize=(10, 3), save_folder=save_folder)

    # Timing
    timing_scores = {}
    for key in evaluate_keys:
        timing_scores[key] = {'Overall': np.mean(get_timing(results[key]['Predicted'], anomaly_start_end))}
        for label in data_label_hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            _indices_of_ones = [index for index, value in enumerate(results['Actual'][mask]) if value == 1]
            _, _anomaly_start_end = find_unterrupted_sequences(_indices_of_ones, 0)
            timing_scores[key][label] = np.mean(get_timing(results[key]['Predicted'][mask], _anomaly_start_end))
    timing_df = pd.DataFrame(timing_scores)
    timing_df.loc['Average'] = timing_df.loc[data_label_hue_order[1:]].mean()
    timing_df = timing_df.loc[row_order]
    visualize_metric_matrix('Timing', timing_df, 'Greens_r', 0, suffix=' min.', high_best=False, figsize=(10, 3), save_folder=save_folder)



