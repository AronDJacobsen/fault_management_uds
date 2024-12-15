
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader



def get_predictions(model, dataset, scalers, configs, steps_ahead):
    # Set model to evaluation mode
    model.eval()
    dataset_args = configs['dataset_args']
    batch_size = configs['training_args']['batch_size']
    sequence_length = dataset_args['sequence_length']
    exogenous_idx = dataset.exogenous_idx
    endogenous_idx = dataset.endogenous_idx
    
    # Pre-allocate
    n_obs = len(dataset)
    n_features = len(endogenous_idx)
    predictions = np.zeros((n_obs, n_features, steps_ahead))
    targets = np.zeros((n_obs, n_features, steps_ahead))

    # Iterate over the dataset
    with torch.no_grad():
        for i, valid_idx in tqdm(enumerate(dataset.valid_indices), desc='Predicting', total=n_obs):

            # Get the model's prediction
            model_output, true_output = predict(model, i, valid_idx, dataset, steps_ahead)
            predictions[i] = model_output
            targets[i] = true_output

    # Inverse transform the data
    predictions = inverse_transform(predictions, scalers, dataset.endogenous_vars, steps_ahead)
    targets = inverse_transform(targets, scalers, dataset.endogenous_vars, steps_ahead)
    return predictions, targets



def predict(model, i, valid_idx, dataset, steps_ahead, scalers=None):
    model.eval()
    # Pre-allocation
    predictions = np.zeros((len(dataset.endogenous_idx), steps_ahead))
    # Get the input sequence
    X = dataset[i][0]
    input_seq = X.unsqueeze(0).contiguous()
    # Get the true steps ahead
    true_ahead = dataset.data[valid_idx:valid_idx+steps_ahead]
    for step in range(steps_ahead):
        # Get the model's prediction
        output = model(input_seq)
        predictions[:, step] = output.cpu().squeeze().detach().numpy()
        # Update the input sequence
        input_seq = torch.cat([input_seq[:, 1:], true_ahead[step].unsqueeze(0).unsqueeze(0)], dim=1) # add batch and seq dims
        # Insert the prediction into the input sequence
        input_seq[:, -1, dataset.endogenous_idx] = output

    targets = true_ahead[:, dataset.endogenous_idx].cpu().squeeze().detach().numpy().T # dim: (endogenous_vars, steps_ahead)

    # Inverse transform the data if scalers are provided
    if scalers:
        # reshape to (steps_ahead, endogenous_vars)
        predictions = predictions.reshape(steps_ahead, len(dataset.endogenous_idx))
        predictions = inverse_transform(predictions, scalers, dataset.endogenous_vars, None)
        targets = targets.reshape(steps_ahead, len(dataset.endogenous_vars))
        targets = inverse_transform(targets, scalers, dataset.endogenous_vars, None)

    return predictions, targets



def inverse_transform(data, scalers, endogenous_vars, steps_ahead=None):
    # Apply inverse transformation using scalers to restore original data values.
    
    if steps_ahead is None:  # Handle 2D data: (observations, variables)
        for var_idx, var_name in enumerate(endogenous_vars):
            data[:, var_idx] = scalers[var_name].inverse_transform(
                data[:, var_idx].reshape(-1, 1)
            ).flatten()
    
    else:  # Handle 3D data: (observations, variables, steps_ahead)
        for var_idx, var_name in enumerate(endogenous_vars):
            # Apply inverse transform for each variable across all steps
            # Reshape the data for the scaler, then reshape back to the original format
            data[:, var_idx, :] = scalers[var_name].inverse_transform(
                data[:, var_idx, :].reshape(-1, 1)
            ).reshape(data.shape[0], steps_ahead)
    return data



# Dataloader
def get_dataloader_predictions(model, dataset, scalers, configs, steps_ahead, num_workers=0):
    # Set model to evaluation mode
    model.eval()
    batch_size = configs['training_args']['batch_size']
    n_obs = len(dataset)  # Total number of observations
    n_features = len(dataset.endogenous_idx)

    # Pre-allocate
    predictions = np.zeros((n_obs, n_features, steps_ahead))
    targets = np.zeros((n_obs, n_features, steps_ahead))

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    # Iterate over the DataLoader
    with torch.no_grad():
        start = 0
        for batch in tqdm(dataloader, desc='Predicting'):
            # Get valid indices for the batch
            valid_idx_batch = dataset.valid_indices[start:start + len(batch[0])]

            # Batch predictions
            batch_preds, batch_targets = batch_predict(model, batch, valid_idx_batch, dataset, steps_ahead)

            # Assign batch predictions to the pre-allocated arrays
            end = start + len(batch_preds)
            predictions[start:end] = batch_preds
            targets[start:end] = batch_targets

            start = end  # Update start index for the next batch

    # Inverse transform the data
    predictions = inverse_transform(predictions, scalers, dataset.endogenous_vars, steps_ahead)
    targets = inverse_transform(targets, scalers, dataset.endogenous_vars, steps_ahead)
    return predictions, targets


def batch_predict(model, batch, valid_idx_batch, dataset, steps_ahead):
    model.eval()

    # Determine batch size and features
    batch_size = len(batch[0])  # Assuming batch[0] contains the input sequences
    n_features = len(dataset.endogenous_idx)

    # Pre-allocate predictions and targets
    predictions = np.zeros((batch_size, n_features, steps_ahead))
    targets = np.zeros((batch_size, n_features, steps_ahead))

    # Prepare input sequences and true targets
    input_seqs = batch[0]  # Assuming batch[0] is the input sequence
    true_ahead = [dataset.data[valid_idx:valid_idx + steps_ahead] for valid_idx in valid_idx_batch]
    true_ahead = torch.stack(true_ahead).to(input_seqs.device)

    # Generate predictions for each step
    for step in range(steps_ahead):
        # Get model output
        output = model(input_seqs)  # Shape: (batch_size, n_features)
        output = output.cpu().detach().numpy()  # Convert to numpy

        # Assign predictions
        predictions[:, :, step] = output

        # Update input sequence for next step
        input_seqs = torch.cat([input_seqs[:, 1:], true_ahead[:, step].unsqueeze(1)], dim=1)
        input_seqs[:, -1, dataset.endogenous_idx] = torch.tensor(output, device=input_seqs.device)

    # Extract targets
    selected_features = true_ahead[:, :, dataset.endogenous_idx]

    if selected_features.ndim == 3:
        targets = selected_features.cpu().numpy().transpose(0, 2, 1)
    else:
        raise ValueError(f"Unexpected shape for selected_features: {selected_features.shape}")

    return predictions, targets


# def get_dataloader_predictions(model, dataset, scalers, configs, steps_ahead):
#     model.eval()
#     batch_size = configs['training_args']['batch_size']
#     n_obs = len(dataset)
#     n_features = len(dataset.endogenous_idx)
    
#     # Pre-allocate predictions and targets
#     predictions = torch.zeros((n_obs, n_features, steps_ahead))
#     targets = torch.zeros((n_obs, n_features, steps_ahead))
    
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
#     with torch.no_grad():
#         start = 0
#         for batch in tqdm(dataloader, desc='Predicting'):
#             valid_idx_batch = dataset.valid_indices[start:start + len(batch[0])]
#             batch_preds, batch_targets = batch_predict(model, batch, valid_idx_batch, dataset, steps_ahead)
            
#             end = start + len(batch_preds)
#             predictions[start:end] = torch.tensor(batch_preds)
#             targets[start:end] = torch.tensor(batch_targets)
#             start = end
    
#     predictions = inverse_transform(predictions.numpy(), scalers, dataset.endogenous_vars, steps_ahead)
#     targets = inverse_transform(targets.numpy(), scalers, dataset.endogenous_vars, steps_ahead)
#     return predictions, targets



# def batch_predict(model, batch, valid_idx_batch, dataset, steps_ahead):
#     model.eval()
    
#     batch_size = len(batch[0])  # Assuming batch[0] contains input sequences
#     n_features = len(dataset.endogenous_idx)
    
#     # Pre-allocate predictions and targets in Torch
#     predictions = torch.zeros((batch_size, n_features, steps_ahead), device=batch[0].device)
#     targets = torch.zeros((batch_size, n_features, steps_ahead), device=batch[0].device)
    
#     input_seqs = batch[0]
#     true_ahead = [dataset.data[valid_idx:valid_idx + steps_ahead] for valid_idx in valid_idx_batch]
#     true_ahead = torch.stack(true_ahead).to(input_seqs.device)
    
#     for step in range(steps_ahead):
#         output = model(input_seqs)  # Predict for the current step
#         predictions[:, :, step] = output
        
#         # Update input_seqs for next step
#         if step < steps_ahead - 1:
#             # Shift input_seqs and append predictions/targets
#             input_seqs = torch.cat([input_seqs[:, 1:], true_ahead[:, step].unsqueeze(1)], dim=1)
#             input_seqs[:, -1, dataset.endogenous_idx] = output

#     # Extract targets
#     selected_features = true_ahead[:, :, dataset.endogenous_idx]
#     targets[:, :, :] = selected_features.transpose(1, 2)
    
#     return predictions.cpu().numpy(), targets.cpu().numpy()




