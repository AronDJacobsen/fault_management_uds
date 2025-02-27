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
from sklearn.metrics import roc_curve


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

    # Update: Lowercase the keys
    column_2_idx = {key.lower(): value for key, value in column_2_idx.items()}


    # Handle features
    remove_last = 0
    if 'final hidden' in column_2_idx:
        # Shift it by 1 and remove the last one
        final_hidden_idx = column_2_idx['final hidden']
        outputs[:, final_hidden_idx] = np.roll(outputs[:, final_hidden_idx], -1) # shift by 1 back
        remove_last = max(remove_last, 1)

    if 'ig' in column_2_idx:
        # Shift it by 1 and remove the last one
        integrated_gradients_idx = column_2_idx['ig']
        outputs[:, integrated_gradients_idx] = np.roll(outputs[:, integrated_gradients_idx], -1) # shift by 1 back
        remove_last = max(remove_last, 1)

    # remove the last one
    outputs = outputs[:-remove_last]
    print(f"Outputs after features: {outputs.shape}")
    return outputs, column_2_idx


# def add_steps_ahead(save_path, outputs, column_2_idx):
#     # Add the steps ahead residuals
#     # Load
#     #save_path = MODELS_DIR / model_save_path / "1_split/evaluation" / data_type / "output.pkl"
#     n_steps_preds = pickle.load(open(save_path, 'rb'))
#     print(n_steps_preds.keys()) 
#     steps_ahead = n_steps_preds['predictions'].shape[2] # dims: (n_samples, n_features, n_steps)
#     print(steps_ahead)
#     # Extract all step predictions
#     all_step_preds = n_steps_preds['predictions']  # Shape: (n_samples, n_features, n_steps)
#     timestamps = pd.to_datetime(n_steps_preds['timestamps'])
#     # create a dataframe based on timestamps as index
#     start, end = timestamps[0], timestamps[-1] + pd.Timedelta(minutes=steps_ahead)
#     full_range = pd.date_range(start=start, end=end, freq='min')

#     df = pd.DataFrame(index=full_range)

#     for step_ahead in range(steps_ahead):
#         # add the minute to the timestamps
#         _timestamps = timestamps + pd.Timedelta(minutes=step_ahead)
#         # insert the predictions into the dataframe
#         df[f"Step {step_ahead}"] = np.nan
#         df.loc[_timestamps, f"Step {step_ahead}"] = all_step_preds[:, :, step_ahead].flatten()

#     # if a row has any nan value, remove it
#     df = df.dropna()
#     print(f"Difference between the outputs and the steps ahead: {outputs.shape[0] - df.shape[0]}")

#     # Filter and match
#     starttimes = pd.to_datetime(outputs[:, column_2_idx['Starttime']].flatten())
#     mask = df.index.isin(starttimes)
#     df = df[mask]
#     steps_to_outputs_idxs = np.searchsorted(starttimes, df.index)
#     # Filter the outputs
#     outputs = outputs[steps_to_outputs_idxs]
#     # Get the residuals
#     residuals = outputs[:, column_2_idx['Target']].reshape(-1,1) - df.values
    
#     # Update outputs with residuals for all lags
#     lag_indices = [outputs.shape[1] + lag for lag in range(steps_ahead)]
#     # add the residuals to the outputs
#     outputs = np.hstack([outputs, residuals])
#     column_2_idx['Residuals'] = lag_indices
#     print(f"Outputs after steps ahead: {outputs.shape}")

#     return outputs, column_2_idx


def get_features(outputs, column_2_idx):
    
    #feature_columns = ['Target', 'Residuals', 'Final hidden', 'IG', 'PIG',] # don't use the 'residual'
    feature_columns = ['Target', 'Final Hidden', 'Residuals', 'IG']#, 'Multi-Feature']

    feature_columns = [x for x in feature_columns if x.lower() in column_2_idx]
    #feature_2_idx = {k: column_2_idx[k.lower()] for k in feature_columns}
    all_feature_indices = []
    #feature_idx_names = []
    for k in feature_columns:
        all_feature_indices.extend(column_2_idx[k.lower()])

        #feature_idx_names.extend([k + '_' + str(len(feature_2_idx[k])-i) if len(feature_2_idx[k]) > 1 else k for i in range(len(feature_2_idx[k]))])


    # Add the Multi-Feature
    feature_columns.append('Multi-Feature')
    column_2_idx['multi-feature'] = all_feature_indices
    #feature_idx_names.append('Multi-Feature')
    
    print(f"Feature columns: {feature_columns}")


    # get the coloring variable
    data_label = outputs[:, column_2_idx['data label']].flatten()
    # convert to label using indicator_2_data_label
    data_label = [indicator_2_data_label[str(int(x))] for x in data_label]

    return feature_columns, column_2_idx, data_label



def get_anomaly_detection_results(models, outputs, column_2_idx, feature_columns, data_label):

    results = {
        'Valid index': outputs[:, column_2_idx['valid index']].flatten(),
        'Starttime': outputs[:, column_2_idx['starttime']].flatten(),
        'Data label': data_label,
        'Actual': (outputs[:, column_2_idx['data label']].flatten()!=0).astype(int),
    }

    # NOTE: IsolationForest is the only applicable model
    model_name = 'IsolationForest' # ['IsolationForest', 'OneClassSVM', 'LOF']



    # Initialize models
    if models is None:
        models = {k: None for k in feature_columns}
    # evaluate_2_idx = feature_2_idx
    # evaluate_2_idx["Multi-Feature"] = all_feature_indices

    # Now generate results for each feature column
    for feature in feature_columns:
        feature_indices = column_2_idx[feature.lower()]
        predicted_anomalies, decision_function, models[feature] = detect_anomalies(model_name, models[feature], outputs[:, feature_indices])
        results[feature] = {
            'Predicted': predicted_anomalies,
            'Decision Function': decision_function,
        }

    return results, models



def save_results(save_folder, results, final_method_selection):
    # final_results = {
    #     # useful for splitting data
    #     '1': results['Valid index'][results[final_method_selection]['Predicted'] == 1],
    #     '0': results['Valid index'][results[final_method_selection]['Predicted'] == 0],
    #     'true_1': results['Valid index'][results['Actual'] == 1],
    #     'true_0': results['Valid index'][results['Actual'] == 0],

    #     'starttimes': results['Starttime'],
    #     'valid_index': results['Valid index'], 

    #     'final_method_selection': final_method_selection,

    #     # decision function
    #     'decision_function': results[final_method_selection]['Decision Function'],
    #     'predicted': results[final_method_selection]['Predicted'],
    #     'actual': results['Actual'], # actual anomaly
    #     'data_label': results['Data label'], # actual data label

    # }
    # # Save the results
    # with open(save_folder / 'anomaly_prediction_results.pkl', 'wb') as f:
    #     pickle.dump(final_results, f)

    # print(f"Results saved at {save_folder / 'anomaly_prediction_results.pkl'}")

    # and save the complete results
    results['final_method_selection'] = final_method_selection
    results['1'] = results['Valid index'][results[final_method_selection]['Predicted'] == 1]
    results['0'] = results['Valid index'][results[final_method_selection]['Predicted'] == 0]


    with open(save_folder / 'anomaly_prediction_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"Full results saved at {save_folder / 'anomaly_prediction_results.pkl'}")



def run_anomaly_detection(models, data_type, final_method_selection, anomalous_path, outputs, column_2_idx):
    save_folder = anomalous_path / data_type
    
    # get the features
    feature_columns, column_2_idx, data_label = get_features(outputs, column_2_idx)

    # # Visualize PCA
    # pca_df, explained_variance, loadings = fit_pca(outputs, all_feature_indices, feature_idx_names, data_label)
    # visualize_pca(save_folder, pca_df, explained_variance, loadings, 20)

    # Anomaly detection
    results, models = get_anomaly_detection_results(models, outputs, column_2_idx, feature_columns, data_label)


    # # Visualize the results
    # auc_scores = cm_roc_auc_results(save_folder, results, evaluate_keys, data_label)
    # metric_results(save_folder, auc_scores, results, evaluate_keys, data_label)

    # # Visualize t-SNE
    # tsne_df = fit_tsne(outputs, all_feature_indices, feature_idx_names, data_label, results[final_method_selection]['Predicted'])
    # visualize_tsne(save_folder, tsne_df)


    # Extract the final results
    save_results(save_folder, results, final_method_selection)

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

    # # Ensure data types are in the correct order
    # dt_order = ['train', 'val', 'test']
    # data_types = [dt for dt in dt_order if dt in data_types]

    models = None # to store the models
    final_method_selection = "Multi-Feature"

    # train the anomaly detection on training set, and apply it on all
    if 'train' in data_types:
        print("Train outputs available")
        # train the model and get results
        print("Training the model") if models is None else print("Using previous model")
        save_folder = anomalous_path / 'train'
        outputs, column_2_idx = load_model_outputs(save_folder)
        #outputs, column_2_idx = add_steps_ahead(evaulation_path / 'train' / 'output.pkl', outputs, column_2_idx)
        # get the features
        models = run_anomaly_detection(models, 'train', final_method_selection, anomalous_path, outputs, column_2_idx)
        print("")

    if 'val' in data_types:
        print("Validation outputs available")
        print("Training the model") if models is None else print("Using previous model")
        save_folder = anomalous_path / 'val'
        outputs, column_2_idx = load_model_outputs(save_folder)
        #outputs, column_2_idx = add_steps_ahead(evaulation_path / 'val' / 'output.pkl', outputs, column_2_idx)
        # get the features
        models = run_anomaly_detection(models, 'val', final_method_selection, anomalous_path, outputs, column_2_idx)
        print("")

    if 'test' in data_types:
        print("Test outputs available")
        print("Training the model") if models is None else print("Using previous model")
        save_folder = anomalous_path / 'test'
        outputs, column_2_idx = load_model_outputs(save_folder)
        #outputs, column_2_idx = add_steps_ahead(evaulation_path / 'test' / 'output.pkl', outputs, column_2_idx)
        # get the features
        models = run_anomaly_detection(models, 'test', final_method_selection, anomalous_path, outputs, column_2_idx)
        print("")

    # TODO: update the thresholds
    # ensure val is in the data_types
    if 'val' in data_types:
        print("Updating thresholds")


        # Load the results
        with open(anomalous_path / "val" / 'anomaly_prediction_results.pkl', 'rb') as f:
            results = pickle.load(f)

        methods = ['Target', 'Final Hidden', 'Residuals', 'IG', 'Multi-Feature']
        # create thresholds
        optimal_thresholds = {}
        for i, key in enumerate(methods):
            # ROC curve
            fpr, tpr, thresholds = roc_curve(results['Actual'], results[key]['Decision Function'])

            # Don't use the first and last threshold
            fpr = fpr[1:-1]
            tpr = tpr[1:-1]
            thresholds = thresholds[1:-1]

            # Calculate Youden's index
            youden_index = tpr - fpr
            optimal_threshold = thresholds[np.argmax(youden_index)]

            print(f'Optimal Threshold for {key}: {optimal_threshold}')
            optimal_thresholds[key] = optimal_threshold

        # Save the optimal thresholds
        with open(anomalous_path / 'optimal_thresholds.pkl', 'wb') as f:
            pickle.dump(optimal_thresholds, f)

        # update the anomaly detection results
        update_thresholds(methods, 'train', anomalous_path)
        update_thresholds(methods, 'val', anomalous_path)
        update_thresholds(methods, 'test', anomalous_path)

        # for method in methods:
        #     results[method]['Predicted'] = (results[method]['Decision Function'] > optimal_thresholds[method]).astype(int)
        # # Update the final method selection results
        # final_method_selection = results['final_method_selection']
        # results['1'] = results['Valid index'][results[final_method_selection]['Predicted'] == 1]
        # results['0'] = results['Valid index'][results[final_method_selection]['Predicted'] == 0]

    else:
        print("Validation data not available. Skipping threshold update")




def update_thresholds(methods, data_type, anomalous_path):

    # load the results
    with open(anomalous_path / data_type / 'anomaly_prediction_results.pkl', 'rb') as f:
        results = pickle.load(f)

    # load the optimal thresholds (ensure path exists)
    with open(anomalous_path / 'optimal_thresholds.pkl', 'rb') as f:
        optimal_thresholds = pickle.load(f)


    # update the anomaly detection results
    for method in methods:
        results[method]['Predicted'] = (results[method]['Decision Function'] > optimal_thresholds[method]).astype(int)
        # Save the theshold
        results[method]['Optimal Threshold'] = optimal_thresholds[method]
    
    # Update the final method selection results
    final_method_selection = results['final_method_selection']
    results['1'] = results['Valid index'][results[final_method_selection]['Predicted'] == 1]
    results['0'] = results['Valid index'][results[final_method_selection]['Predicted'] == 0]

    # Save the results
    with open(anomalous_path / data_type / 'anomaly_prediction_results.pkl', 'wb') as f:
        pickle.dump(results, f)





if __name__ == '__main__':
    main()










