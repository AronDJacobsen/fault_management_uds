import json
import pickle
import itertools
import os
import argparse

import numpy as np




from fault_management_uds.config import PROJ_ROOT
from fault_management_uds.config import DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR
from fault_management_uds.config import MODELS_DIR, REPORTS_DIR, FIGURES_DIR, REFERENCE_DIR
from fault_management_uds.config import rain_gauge_color, condition_to_meta


from fault_management_uds.get_detection import load_model_outputs, run_anomaly_detection, get_features, update_thresholds
from fault_management_uds.get_features import add_steps_ahead
from sklearn.metrics import roc_curve



def parse_args():
    parser = argparse.ArgumentParser(description='Fault Management UDS')
    parser.add_argument('--iteration', type=int, default=1, help='Iterative run number')
    parser.add_argument('--fast_run', type=bool, default=False, help='Quick run')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    return parser.parse_args()





def main():
    ### Set up
    # Parse arguments
    args = parse_args()
    iteration = args.iteration
    n_runs = 2 ** int(iteration)

    # Only interested in the test data
    #data_type = "test"

    # load iterations folders
    save_folder = "transformer/7_anomalous"
    save_folder = MODELS_DIR / save_folder
    
    prefix = "iteration="
    ano_relative_path = "1_split/anomalous"
    eval_relative_path = "1_split/evaluation"

    # Have to run test to the get the model outputs
    models = None
    data_types = ["train", "val", "test"]   
    for data_type in data_types:
        # Find all model outputs
        all_runs = os.listdir(save_folder)
        all_runs = [run for run in all_runs if run.startswith(prefix)]
        runs = []
        for run in all_runs:
            # get the iteration number
            iteration_identifier = run.split("=")[-1].split("_")[0]
            n_iteration = len(iteration_identifier.split(".")) - 1
            if int(n_iteration) == int(iteration):
                runs.append(run)
        assert len(runs) == n_runs, f"Expected {n_runs} runs, but found {len(runs)}"


        # Collect all model outputs into one
        # Iterate over all model outputs, col 2 idx is same for all
        model_outputs = []
        for run in runs:
            # Load outputs (multiple)
            outputs_folder = save_folder / run / ano_relative_path / data_type
            outputs, column_2_idx = load_model_outputs(outputs_folder)
            feature_columns, column_2_idx, data_label = get_features(outputs, column_2_idx)


            # outputs_folder = save_folder / run / ano_relative_path / data_type
            # outputs, column_2_idx = load_model_outputs(outputs_folder)
            # # Load steps ahead output
            # evaluation_folder = save_folder / run / eval_relative_path / data_type
            # outputs, column_2_idx = add_steps_ahead(evaluation_folder / 'output.pkl', outputs, column_2_idx)
            model_outputs.append(outputs)
        
        # Combine all model outputs
        # Ensure all outputs have the same number of columns before concatenating
        if len(model_outputs) > 0:
            reference_shape = model_outputs[0].shape[1]
            assert all(m.shape[1] == reference_shape for m in model_outputs), "Mismatch in column count across runs"

        # Combine all model outputs
        combined_outputs = np.concatenate(model_outputs, axis=0)

        # Run the evaluation
        #final_feature_selection = 
        final_feature_selection = "Multi-Feature"
        # Create a new path for the anomalous data
        anomalous_path = save_folder / f"combined_iteration={iteration}"
        full_path = anomalous_path / data_type
        full_path.mkdir(parents=True, exist_ok=True)
        models = run_anomaly_detection(models, data_type, final_feature_selection, anomalous_path, combined_outputs, column_2_idx)



    # TODO: update the thresholds
    # ensure val is in the data_types
    if 'val' in data_types:
        print("Updating thresholds")


        # Load the results
        with open(anomalous_path / "val" / 'anomaly_prediction_results.pkl', 'rb') as f:
            results = pickle.load(f)

        methods = ['Multi-Feature']
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






if __name__ == '__main__':
    main()




