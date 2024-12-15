

### Evaluation

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import pickle


from fault_management_uds.modelling.predict import get_predictions, get_dataloader_predictions
from fault_management_uds.data.dataset import load_conditions


def evaluate_model_on_dataset(eval_folder, model, dataset, scalers, configs, data_type='test'):
    eval_folder.mkdir(exist_ok=True)
    
    ### Get predictions
    #predictions, targets = get_predictions(model, dataset, scalers, configs.config, configs.config['predict_steps_ahead'])
    predictions, targets = get_dataloader_predictions(model, dataset, scalers, configs.config, configs.config['predict_steps_ahead'], configs.num_workers)

    # Save
    output = {
        'predictions': predictions,
        'targets': targets,
        'endogenous_vars': dataset.endogenous_vars,
        'timestamps': dataset.valid_timestamps,
    }
    with open(eval_folder / f'output.pkl', 'wb') as f:
        pickle.dump(output, f)

    ### Evaluate
    # Get the MAEs for each step ahead
    MAEs = evaluate_step_predictions(predictions, targets, dataset.endogenous_vars, configs.config['predict_steps_ahead'])
    # save the MAEs
    MAEs.to_csv(eval_folder / f'step_MAEs.csv', index=False)

    # Get the conditions
    conditions = load_conditions(dataset.valid_timestamps[0], dataset.valid_timestamps[-1])
    # Evaluate the model on the conditions
    MAEs = evaluate_condition_predictions(predictions, targets, dataset, configs.config['predict_steps_ahead'], conditions)
    # save
    MAEs.to_csv(eval_folder / 'condition_MAEs.csv')




def evaluate_mae(predictions, targets):
    # Calculate the mean absolute error
    assert predictions.shape == targets.shape, "Predictions and targets must have the same shape"
    mae = np.sum(np.abs(predictions - targets)) / len(predictions)
    return mae


def evaluate_step_predictions(predictions, targets, endogenous_vars, steps_ahead):
    MAEs = {'Step': list(range(1, steps_ahead+1)), 'Overall': []}

    # for each variable
    for i, endogenous_var in enumerate(endogenous_vars):
        MAEs[endogenous_var] = [None] * steps_ahead  

        # for each step ahead
        for step in range(steps_ahead):
            mae = evaluate_mae(predictions[:, i, step], targets[:, i, step])
            MAEs[endogenous_var][step] = mae

    # Overall is the average of all the dependent variables
    for step in range(steps_ahead):
        overall_mae = np.mean([MAEs[endogenous_var][step] for endogenous_var in endogenous_vars])
        MAEs['Overall'].append(overall_mae)    

    MAEs = pd.DataFrame(MAEs)

    return MAEs


def evaluate_condition_predictions(predictions, targets, dataset, steps_ahead, conditions):

    # Initialize an empty list to collect results
    results = []

    # Loop through each condition
    for condition_name, condition_data in conditions.items():
        # Extract the relevant indices
        relevant_timestamps = np.intersect1d(dataset.valid_timestamps, condition_data['timestamps'])
        # get their indices
        relevant_indices = np.where(np.isin(dataset.valid_timestamps, relevant_timestamps))[0]

        for i, endogenous_var in enumerate(dataset.endogenous_vars):
            for step in range(steps_ahead):
                # Filter target, mask, and predictions
                target = targets[relevant_indices, i, step]
                prediction = predictions[relevant_indices, i, step]

                # Evaluate MAE
                mae = evaluate_mae(prediction, target)

                # Collect the results
                results.append({
                    'Condition': condition_name,
                    'Step': step + 1,
                    'Sensor': endogenous_var,
                    'MAE': mae,
                })

    # dataframe
    MAEs = pd.DataFrame(results)
    # set step and condition to multi-index
    MAEs = MAEs.set_index(['Step', 'Condition'])
    # set sensor as columns
    MAEs = MAEs.pivot(columns='Sensor', values='MAE')
    # sort index by conditions
    MAEs = MAEs.reindex(conditions.keys(), level=1)

    # calculate the overall MAE
    MAEs['Overall'] = MAEs.mean(axis=1)
    MAEs = MAEs[['Overall'] + [col for col in MAEs.columns if col != 'Overall']]

    return MAEs

