import json
import pickle
import itertools
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from fault_management_uds.plots import visualize_pca, cm_roc_auc_results, metric_results, fit_pca, fit_tsne, visualize_tsne
from fault_management_uds.modelling.classifiers import detect_anomalies


from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR
from fault_management_uds.config import rain_gauge_color, condition_to_meta


from fault_management_uds.data.load import import_metadata


# load
indicator_2_data_label = json.load(open(REFERENCE_DIR / 'indicator_2_data_label.json', 'r'))



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--model_save_path', type=str, default='default.yaml', help='Model save path')
    parser.add_argument('--data_group', type=str, default='anomalous', help='Data group to run on')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()



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
    steps_ahead = n_steps_preds['predictions'].shape[2] # dims: (n_samples, n_features, n_steps)
    print(steps_ahead)
    # Extract all step predictions
    all_step_preds = n_steps_preds['predictions']  # Shape: (n_samples, n_features, n_steps)
    timestamps = pd.to_datetime(n_steps_preds['timestamps'])
    # create a dataframe based on timestamps as index
    start, end = timestamps[0], timestamps[-1] + pd.Timedelta(minutes=steps_ahead)
    full_range = pd.date_range(start=start, end=end, freq='min')

    df = pd.DataFrame(index=full_range)

    for step_ahead in range(steps_ahead):
        # add the minute to the timestamps
        _timestamps = timestamps + pd.Timedelta(minutes=step_ahead)
        # insert the predictions into the dataframe
        df[f"Step {step_ahead}"] = np.nan
        df.loc[_timestamps, f"Step {step_ahead}"] = all_step_preds[:, :, step_ahead].flatten()

    # if a row has any nan value, remove it
    df = df.dropna()
    print(f"Difference between the outputs and the steps ahead: {outputs.shape[0] - df.shape[0]}")

    # Filter and match
    starttimes = pd.to_datetime(outputs[:, column_2_idx['Starttime']].flatten())
    mask = df.index.isin(starttimes)
    df = df[mask]
    steps_to_outputs_idxs = np.searchsorted(starttimes, df.index)
    # Filter the outputs
    outputs = outputs[steps_to_outputs_idxs]
    # Get the residuals
    residuals = outputs[:, column_2_idx['Target']].reshape(-1,1) - df.values
    
    # Update outputs with residuals for all lags
    lag_indices = [outputs.shape[1] + lag for lag in range(steps_ahead)]
    # add the residuals to the outputs
    outputs = np.hstack([outputs, residuals])
    column_2_idx['Residuals'] = lag_indices
    print(f"Outputs after steps ahead: {outputs.shape}")

    return outputs, column_2_idx


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

    return feature_columns, feature_2_idx, all_feature_indices, feature_idx_names, data_label



def get_anomaly_detection_results(models, outputs, column_2_idx, feature_columns, feature_2_idx, all_feature_indices, data_label):
    results = {
        'Valid index': outputs[:, column_2_idx['Valid index']].flatten(),
        'Starttime': outputs[:, column_2_idx['Starttime']].flatten(),
        'Data label': data_label,
        'Actual': (outputs[:, column_2_idx['Data label']].flatten()!=0).astype(int),
    }

    # NOTE: IsolationForest is the only applicable model
    model_name = 'IsolationForest' # ['IsolationForest', 'OneClassSVM', 'LOF']

    evaluate_keys = ['Combined'] + feature_columns
    if models is None:
        models = {k: None for k in evaluate_keys}
    evaluate_2_idx = feature_2_idx
    evaluate_2_idx['Combined'] = all_feature_indices

    # Now generate results for each feature column
    for feature in evaluate_keys:
        feature_indices = evaluate_2_idx[feature]
        predicted_anomalies, decision_function, models[feature] = detect_anomalies(model_name, models[feature], outputs[:, feature_indices])
        results[feature] = {
            'Predicted': predicted_anomalies,
            'Decision Function': decision_function,
        }

    return results, evaluate_keys, models



def save_results(save_folder, results, final_feature_selection):
    final_results = {
        # useful for splitting data
        '1': results['Valid index'][results[final_feature_selection]['Predicted'] == 1],
        '0': results['Valid index'][results[final_feature_selection]['Predicted'] == 0],
        'true_1': results['Valid index'][results['Actual'] == 1],
        'true_0': results['Valid index'][results['Actual'] == 0],

        'starttimes': results['Starttime'],
        'valid_index': results['Valid index'], 

        # decision function
        'decision_function': results[final_feature_selection]['Decision Function'],
        'predicted': results[final_feature_selection]['Predicted'],
        'data_label': results['Data label'], # actual data label
        'actual': results['Actual'], # actual anomaly
    }
    # Save the results
    with open(save_folder / 'anomaly_prediction_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)

    print(f"Results saved at {save_folder / 'anomaly_prediction_results.pkl'}")


def run_anomaly_detection(models, data_type, final_feature_selection, anomalous_path, evaulation_path):
    save_folder = anomalous_path / data_type
    
    # load and prepare data
    outputs, column_2_idx = load_model_outputs(save_folder)
    outputs, column_2_idx = add_steps_ahead(evaulation_path / data_type / 'output.pkl', outputs, column_2_idx)
    
    # get the features
    feature_columns, feature_2_idx, all_feature_indices, feature_idx_names, data_label = get_features(outputs, column_2_idx)

    # Visualize PCA
    pca_df, explained_variance, loadings = fit_pca(outputs, all_feature_indices, feature_idx_names, data_label)
    visualize_pca(save_folder, pca_df, explained_variance, loadings, 20)

    # Anomaly detection
    results, evaluate_keys, models = get_anomaly_detection_results(models, outputs, column_2_idx, feature_columns, feature_2_idx, all_feature_indices, data_label)

    # Visualize the results
    auc_scores = cm_roc_auc_results(save_folder, results, evaluate_keys, data_label)
    metric_results(save_folder, auc_scores, results, evaluate_keys, data_label)

    # Visualize t-SNE
    tsne_df = fit_tsne(outputs, all_feature_indices, feature_idx_names, data_label, results[final_feature_selection]['Predicted'])
    visualize_tsne(save_folder, tsne_df)

    # Extract the final results
    save_results(save_folder, results, final_feature_selection)

    return models



        




def main():
    ### Set up
    # Parse arguments
    args = parse_args()

    # prepare
    model_save_path = args.model_save_path
    anomalous_path = MODELS_DIR / model_save_path / '1_split' / 'anomalous'
    evaulation_path = MODELS_DIR / model_save_path / '1_split' / 'evaluation'
    data_types = os.listdir(anomalous_path)
    print(f"Data types available: {data_types}")

    models = None
    final_feature_selection = "Combined"

    # train the anomaly detection on training set, and apply it on all
    if 'train' in data_types:
        print("Train outputs available")
        # train the model and get results
        print("Training the model") if models is None else print("Using previous model")
        models = run_anomaly_detection(models, 'train', final_feature_selection, anomalous_path, evaulation_path)
        print("")

    if 'val' in data_types:
        print("Validation outputs available")
        print("Training the model") if models is None else print("Using previous model")
        models = run_anomaly_detection(models, 'val', final_feature_selection, anomalous_path, evaulation_path)
        print("")

    if 'test' in data_types:
        print("Test outputs available")
        print("Training the model") if models is None else print("Using previous model")
        models = run_anomaly_detection(models, 'test', final_feature_selection, anomalous_path, evaulation_path)
        print("")


if __name__ == '__main__':
    main()










