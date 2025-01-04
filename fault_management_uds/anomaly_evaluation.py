import json
import pickle
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from fault_management_uds.synthetic.synthetic_generator import find_unterrupted_sequences

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from fault_management_uds.plots import pca_plot, visualize_metric_matrix, visualize_confusion, visualize_roc_auc
from fault_management_uds.modelling.classifiers import detect_anomalies


from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR
from fault_management_uds.config import rain_gauge_color, condition_to_meta


from fault_management_uds.data.load import import_metadata

hue_map = {
    'Spike': 'OrangeRed',
    'Noise': 'DarkGray',
    'Frozen': 'SteelBlue',
    'Offset': 'Gold',
    'Drift': 'LightSeaGreen',
    'Original': 'blanchedalmond', 
}
hue_order = list(hue_map.keys())[::-1]
# load
indicator_2_data_label = json.load(open(REFERENCE_DIR / 'indicator_2_data_label.json', 'r'))


def load_model_outputs(outputs_path):
    # Loading
    outputs = pickle.load(open(outputs_path / 'outputs.pkl', 'rb'))
    print(f"Outputs: {outputs.shape}")
    column_2_idx = json.load(open(outputs_path / 'column_2_idx.json', 'r'))
    column_2_idx = {(k[0].upper() + k[1:]).replace('_', ' '): v 
                    for k, v in column_2_idx.items()}
    print(f"Column 2 idx: {list(column_2_idx.keys())}")

    # Handle features
    remove_last = 0
    if 'Final hidden' in column_2_idx:
        # Shift it by 1 and remove the last one
        final_hidden_idx = column_2_idx['Final hidden']
        outputs[:, final_hidden_idx] = np.roll(outputs[:, final_hidden_idx], -1) # shift by 1 back
        remove_last = max(remove_last, 1)

    if 'IG' in column_2_idx:
        # Shift it by 1 and remove the last one
        integrated_gradients_idx = column_2_idx['IG']
        outputs[:, integrated_gradients_idx] = np.roll(outputs[:, integrated_gradients_idx], -1) # shift by 1 back
        remove_last = max(remove_last, 1)

    # remove the last one
    outputs = outputs[:-remove_last]
    print(f"Outputs after features: {outputs.shape}")
    return outputs, column_2_idx


def add_steps_ahead(save_path, outputs, column_2_idx):
    # Add the steps ahead residuals
    # Load
    #save_path = MODELS_DIR / model_save_path / "1_split/evaluation" / data_type / "output.pkl"
    n_steps_preds = pickle.load(open(save_path, 'rb'))
    print(n_steps_preds.keys()) 
    steps_ahead = n_steps_preds['predictions'].shape[2]
    print(steps_ahead)
    # Extract all step predictions
    all_step_preds = n_steps_preds['predictions']  # Shape: (n_samples, n_features, n_steps)
    timestamps = pd.to_datetime(n_steps_preds['timestamps'])
    # Time stamps in the outputs data
    starttimes = pd.to_datetime(outputs[:, column_2_idx['Starttime']].flatten())
    # Initialize a container for residuals for each lag
    residuals_list = []
    # Calculate residuals for each lag
    for lag in range(steps_ahead):
        # Get predictions for the current lag
        step_preds = all_step_preds[:, :, lag]
        # Shift timestamps by the current lag
        step_ts = timestamps + pd.Timedelta(minutes=lag)
        # Filter valid timestamps
        mask = step_ts.isin(starttimes)
        valid_step_ts = step_ts[mask]
        valid_step_preds = step_preds[mask]
        # Locate the index in starttimes where it matches valid_step_ts
        indices = np.searchsorted(starttimes, valid_step_ts)
        # Calculate residuals for the current lag
        residual = outputs[indices, column_2_idx['Target']] - valid_step_preds.flatten()
        # Store residuals in the container
        residuals_list.append(residual.reshape(-1, 1))

    # Combine residuals for all lags
    residuals_list = residuals_list[::-1]
    all_residuals = np.hstack(residuals_list)

    # Update outputs with residuals for all lags
    lag_indices = [outputs.shape[1] + lag for lag in range(steps_ahead)]
    outputs = np.hstack([outputs, all_residuals])
    column_2_idx['Residuals'] = lag_indices
    print(f"Outputs after steps ahead: {outputs.shape}")

    return # TODO


def get_features(outputs, column_2_idx):
    feature_columns = ['Target', 'Residuals', 'Final hidden', 'IG', 'PIG',]
    feature_columns = [x for x in feature_columns if x in column_2_idx]
    feature_2_idx = {k: column_2_idx[k] for k in feature_columns}
    all_feature_indices = []
    feature_idx_names = []
    for k in feature_columns:
        all_feature_indices.extend(feature_2_idx[k])

        feature_idx_names.extend([k + '_' + str(len(feature_2_idx[k])-i) if len(feature_2_idx[k]) > 1 else k for i in range(len(feature_2_idx[k]))])

    # get the coloring variable
    data_label = outputs[:, column_2_idx['Data label']].flatten()
    # convert to label using indicator_2_data_label
    data_label = [indicator_2_data_label[str(int(x))] for x in data_label]

    return # TODO

def fit_pca():
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
    # Explained variance
    explained_variance = pca.explained_variance_ratio_

    return # TODO:


def visualize_pca(save_folder):

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

    plot_pcs = list(pca_df.columns[list(range(max_components))])
    plot_pcs = ['PC1', 'PC2', 'PC3', 'PC4']
    pcs_combs = list(itertools.combinations(plot_pcs, 2))

    for x, y in pcs_combs:
        pca_plot(pca_df, x, y, explained_variance, 'Data label', hue_map,
                plot_loadings=False, loadings=None,
                save_folder=save_folder)

    # Loadings
    n_components = idx_95+1 # seem to explain most of the variance
    plt.figure(figsize=(12, 5))
    sns.heatmap(loadings.iloc[:, :n_components].T, annot=False, cmap='coolwarm', center=0, fmt=".2f", cbar=True)
    plt.xlabel('Hidden dimension', fontsize=14)
    # rotate the y labels
    plt.yticks(rotation=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_folder / 'pca_loadings.png')


def get_anomaly_detection_results():
    results = {
        'Valid index': outputs[:, column_2_idx['Valid index']].flatten(),
        'Data label': data_label,
        'Actual': (outputs[:, column_2_idx['Data label']].flatten()!=0).astype(int),
    }

    # NOTE: IsolationForest is the only applicable model
    model_name = 'IsolationForest' # ['IsolationForest', 'OneClassSVM', 'LOF']

    evaluate_keys = ['Combined'] + feature_columns
    evaluate_2_idx = feature_2_idx
    evaluate_2_idx['Combined'] = all_feature_indices

    # Now generate results for each feature column
    for feature in evaluate_keys:
        print(f'Feature: {feature}')
        feature_indices = evaluate_2_idx[feature]
        predicted_anomalies, decision_function = detect_anomalies(model_name, outputs[:, feature_indices])
        results[feature] = {
            'Predicted': predicted_anomalies,
            'Decision Function': decision_function,
        }

    return # TODO



def cm_roc_auc_results(save_folder):
    # Overall results
    fig, axes = plt.subplots(3, len(evaluate_keys), figsize=(15, 7))
    auc_scores = {}
    for i, key in enumerate(evaluate_keys):
        # Confusion matrix
        conf_matrix = confusion_matrix(results['Actual'], results[key]['Predicted'])
        axes[0, i] = visualize_confusion(axes[0, i], i, conf_matrix, 'd', 'Blues')
        # Percentage confusion matrix
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        axes[1, i] = visualize_confusion(axes[1, i], i, conf_matrix, ".2f", 'Blues')

        # ROC curve
        fpr, tpr, _ = roc_curve(results['Actual'], results[key]['Decision Function'])
        roc_auc = auc(fpr, tpr)
        auc_scores[key] = {'Overall': roc_auc}
        axes[2, i] = visualize_roc_auc(axes[2, i], i, fpr, tpr, roc_auc)
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_folder / 'cm_roc_auc_overall.png')


    # For each anomaly
    for label in hue_order[::-1]:
        if label == 'Original':
            continue
        print(f'Anomaly: {label}')
        mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(2, len(evaluate_keys), figsize=(15, 5))
        for i, key in enumerate(evaluate_keys):
            # Confusion matrix
            conf_matrix = confusion_matrix(results['Actual'][mask], results[key]['Predicted'][mask])
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            axes[0, i] = visualize_confusion(axes[0, i], i, conf_matrix, ".2f", 'Blues')

            # ROC curve
            fpr, tpr, _ = roc_curve(results['Actual'][mask], results[key]['Decision Function'][mask])
            roc_auc = auc(fpr, tpr)
            auc_scores[key][label] = roc_auc
            axes[1, i] = visualize_roc_auc(axes[1, i], i, fpr, tpr, roc_auc)

        plt.tight_layout()
        #plt.show()
        plt.savefig(save_folder / f'cm_roc_auc_{label}.png')
        print('\n')

    return auc_scores



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


def metric_results(save_folder):
    # row ordering
    row_order = hue_order[1:] + ['Average', 'Overall']

    # AUC
    auc_df = pd.DataFrame(auc_scores)
    # Calculate the average AUC as well
    auc_df.loc['Average'] = auc_df.loc[hue_order[1:]].mean()
    auc_df = auc_df.loc[row_order]
    visualize_metric_matrix('AUC', auc_df, 'Oranges', 2, suffix=None, figsize=(10, 3), save_folder=save_folder)


    # Precision
    precision_scores = {}
    for key in evaluate_keys:
        precision_scores[key] = {'Overall': precision_score(results['Actual'], results[key]['Predicted'])*100}
        for label in hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            precision_scores[key][label] = precision_score(
                results['Actual'][mask], results[key]['Predicted'][mask]
            ) * 100
    precision_df = pd.DataFrame(precision_scores)
    precision_df.loc['Average'] = precision_df.loc[hue_order[1:]].mean()
    precision_df = precision_df.loc[row_order]
    visualize_metric_matrix('Precision', precision_df, 'Blues', 2, suffix='%', figsize=(10, 3), save_folder=save_folder)


    # Recall
    recall_scores = {}
    for key in evaluate_keys:
        recall_scores[key] = {'Overall': recall_score(results['Actual'], results[key]['Predicted'])*100}
        for label in hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            recall_scores[key][label] = recall_score(
                results['Actual'][mask], results[key]['Predicted'][mask]
            ) * 100
    recall_df = pd.DataFrame(recall_scores)
    recall_df.loc['Average'] = recall_df.loc[hue_order[1:]].mean()
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
        for label in hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            _indices_of_ones = [index for index, value in enumerate(results['Actual'][mask]) if value == 1]
            _, _anomaly_start_end = find_unterrupted_sequences(_indices_of_ones, 0)
            coverage_scores[key][label] = np.mean(get_coverage(results[key]['Predicted'][mask], _anomaly_start_end)) * 100
    coverage_df = pd.DataFrame(coverage_scores)
    coverage_df.loc['Average'] = coverage_df.loc[hue_order[1:]].mean()
    coverage_df = coverage_df.loc[row_order]
    visualize_metric_matrix('Coverage', coverage_df, 'Purples', 2, suffix='%', figsize=(10, 3), save_folder=save_folder)



    # Timing
    timing_scores = {}
    for key in evaluate_keys:
        timing_scores[key] = {'Overall': np.mean(get_timing(results[key]['Predicted'], anomaly_start_end))}
        for label in hue_order:
            if label == 'Original':
                continue
            mask = (np.array(data_label) == label) | (np.array(data_label) == 'Original')
            _indices_of_ones = [index for index, value in enumerate(results['Actual'][mask]) if value == 1]
            _, _anomaly_start_end = find_unterrupted_sequences(_indices_of_ones, 0)
            timing_scores[key][label] = np.mean(get_timing(results[key]['Predicted'][mask], _anomaly_start_end))
    timing_df = pd.DataFrame(timing_scores)
    timing_df.loc['Average'] = timing_df.loc[hue_order[1:]].mean()
    timing_df = timing_df.loc[row_order]
    visualize_metric_matrix('Timing', timing_df, 'Greens_r', 0, suffix=' min.', highest_best=False, figsize=(10, 3), save_folder=save_folder)


def save_results():
    # TODO:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--model_save_path', type=str, default='default.yaml', help='Model save path')
    parser.add_argument('--data_group', type=str, default='anomalous', help='Data group to run on')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()


def main():
    ### Set up
    # Parse arguments
    args = parse_args()

    model_save_path = args.model_save_path

    # load data
    anomalous_path = MODELS_DIR / model_save_path / '1_split' / 'anomalous'

    # load the
    data_types = os.listdir(anomalous_path)

    # train the anomaly detection on training set, and apply it on all


    if 'train' in data_types:
        save_folder = anomalous_path / 'train'
        data = load_data(anomalous_path / 'train')
        anomaly_model = train_anomaly_detection(data)

    if 'val' in data_types:
        # TODO:
        anomaly_model = train_anomaly_detection(data) if 'train' in data_types else anomaly_model
        pass

    if 'test' in data_types:
        pass



if __name__ == '__main__':
    main()










