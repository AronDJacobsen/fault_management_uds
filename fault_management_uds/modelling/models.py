import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from fault_management_uds.utilities import get_accelerator
from fault_management_uds.config import MODELS_DIR

from .loss import get_loss


def load_model_checkpoint(run_folder, run_info, model_to_load, configs, additional_configurations={}):
    # load the model
    checkpoint_path = run_folder / run_info[model_to_load]
    model = get_model(configs['model_args'], configs['training_args'], checkpoint_path, additional_configurations)
    model.to(get_accelerator())
    model.eval() # set to evaluation mode
    print(f"Model loaded from {checkpoint_path}")
    return model



def get_model(model_args, training_args, checkpoint_path=None, additional_configurations={}):
    model_name = model_args['model_name']


    if model_name == 'MeanPredictor':
        model = MeanPredictor(
                        target_mean=additional_configurations['target_mean']
                        )
    
    elif model_name == 'Lag1Predictor':
        model = Lag1Predictor(
                        endogenous_idx=additional_configurations['endogenous_idx']
                        )

    elif model_name == 'LinearRegression':
        model = LinearRegressionModel(
                        input_size=model_args['input_size'],
                        sequence_length=model_args['sequence_length'], 
                        output_size=model_args['output_size']
                        )

    elif model_name == 'MLP':
        model = MLPModel(
                        input_size=model_args['input_size'],
                        sequence_length=model_args['sequence_length'], 
                        hidden_size=model_args['hidden_size'], 
                        output_size=model_args['output_size'],
                        num_layers=model_args['num_layers'], 
                        dropout=model_args['dropout']
                        )

    elif model_name == 'LSTM':
        model = LSTMModel(
                        input_size=model_args['input_size'],
                        sequence_length=model_args['sequence_length'], 
                        hidden_size=model_args['hidden_size'], 
                        output_size=model_args['output_size'],
                        num_layers=model_args['num_layers'], 
                        dropout=model_args['dropout'],
                        predict_difference=model_args['predict_difference'],
                        endogenous_idx=additional_configurations['endogenous_idx'],
                        )

    elif model_name == 'Transformer':
        model = TransformerModel(
                        input_size=model_args['input_size'],
                        sequence_length=model_args['sequence_length'], 
                        hidden_size=model_args['hidden_size'], 
                        use_embedding_layer=model_args['use_embedding_layer'],
                        output_size=model_args['output_size'],
                        num_heads=model_args['num_heads'], 
                        num_layers=model_args['num_layers'], 
                        positional_encoding=model_args['positional_encoding'],
                        dropout=model_args['dropout'],
                        aggregate_hidden=model_args['aggregate_hidden'],
                        predict_difference=model_args['predict_difference'],
                        endogenous_idx=additional_configurations['endogenous_idx'],
                        )

                          
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # use the pre-defined loss function
    criterion = get_loss(training_args['loss_function'])

    # Create a PyTorch Lightning model
    lightning_model = LightningModel(
        model, criterion, 
        learning_rate=float(training_args['lr']), 
        seed=training_args['seed'],
        skip_optimizer=training_args['skip_optimizer'],
        scheduler=training_args['scheduler'],
        epochs=training_args['max_epochs'],
        steps_per_epoch=additional_configurations['n_obs'] // training_args['batch_size'],
        directional_errors=training_args['directional_errors'],
        dynamic_weighting=training_args['dynamic_weighting']
        )

    # Load the model state from the checkpoint if a path is provided
    if checkpoint_path is not None:
        #lightning_model.load_state_dict(torch.load(checkpoint_path)['state_dict'], map_location=get_accelerator())
        lightning_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])


    # if fine_tune_path:
    #     # Load a config file
    #     model_path = MODELS_DIR / fine_tune_path
    #     # check if the folder exists
    #     assert model_path.exists(), f"Model folder {model_path} does not exist."
    #     with open(model_path / 'config.yaml', 'r') as file:
    #         config = yaml.safe_load(file)
    #         model_to_load = config['training_args']['model_to_load']


    return lightning_model



class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for training models.
    """

    def __init__(self, model, criterion, learning_rate=0.001, seed=42, 
                 skip_optimizer=False, scheduler=False, epochs=100, steps_per_epoch=1000,
                 directional_errors=False, dynamic_weighting=False):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'criterion'])
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.seed = seed
        self.skip_optimizer = skip_optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.directional_errors = directional_errors
        self.dynamic_weighting = dynamic_weighting

    def forward(self, x):
        prediction, *_ = self.model(x)
        return prediction


    def handle_dynamic_weighting(self, loss, previous, targets):
        # Calculate the dynamic weighting; assign higher weights to larger differences
        dynamic_weighting = torch.abs(targets - previous)

        # Multiply the loss by the dynamic weighting
        loss = loss * (1 + dynamic_weighting)

        return loss

    def handle_directional_errors(self, loss, previous, targets, outputs):
        # Calculate the directional errors; assign higher weights to incorrect directions
        directional_errors = (torch.sign(targets - previous) != torch.sign(outputs - previous)).float() # 1 if error, 0 if correct

        # Multiply the loss by the weighted directional errors
        loss = loss * (1 + self.directional_errors * directional_errors)

        return loss


    def training_step(self, batch, batch_idx):
        # data: (batch_size, sequence_length, input_size)
        # targets: (batch_size, output_size)
        data, targets, _, _ = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        loss = self.handle_directional_errors(loss, data[:, -1, self.model.endogenous_idx], targets, outputs) if self.directional_errors else loss
        loss = self.handle_dynamic_weighting(loss, data[:, -1, self.model.endogenous_idx], targets) if self.dynamic_weighting else loss
        # Return the loss to scalar
        loss = loss.mean()
        
        self.log('train_loss', loss, prog_bar=True, batch_size=data.size(0), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets, _, _ = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets)
        loss = loss.mean()
        self.log('val_loss', loss, prog_bar=True, batch_size=data.size(0))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.scheduler:
        # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,  # maximum learning rate
                epochs=self.epochs,  # total number of epochs
                steps_per_epoch=self.steps_per_epoch,
                anneal_strategy='cos',  # cosine annealing (default)
                final_div_factor=10000  # reduce learning rate at the end (default)
            )
            return {'optimizer': optimizer, 'scheduler': scheduler}

        # else return the optimizer
        return optimizer


    # def additional_configurations(self, dataset):
    #     # if the model has additional configurations, call the function
    #     if hasattr(self.model, 'additional_configurations'):
    #         self.model.additional_configurations(dataset)
    #     else:
    #         print("Model does not have additional configurations.")


# use pytorch lightining module class as the base class
class MeanPredictor(nn.Module):
    """
    A simple model that predicts the mean of the target variable.
    """

    def __init__(self, target_mean=0.0):   
        super(MeanPredictor, self).__init__()
        self.model_name = 'MeanPredictor'
        # Initialize mean as a parameter
        self.mean = nn.Parameter(torch.tensor(target_mean), requires_grad=False)


    def forward(self, x):
        # input: (batch_size, sequence_length, input_size)

        # get the batch size
        batch_size = x.size(0)
        # repeat the mean value for the batch size
        return torch.full((batch_size, 1), self.mean.item(), device=x.device), None


class Lag1Predictor(nn.Module):
    """
    A simple model that predicts the last value of the target variable.
    """

    def __init__(self, endogenous_idx=[]):
        super(Lag1Predictor, self).__init__()
        self.model_name = 'Lag1Predictor'
        self.endogenous_idx = endogenous_idx


    def forward(self, x):
        # input: (batch_size, sequence_length, input_size)
        return x[:, -1, self.endogenous_idx], None



class LinearRegressionModel(pl.LightningModule):
    """
    Linear regression model for time series forecasting.
    """

    def __init__(self, input_size, sequence_length, output_size):
        super(LinearRegressionModel, self).__init__()
        self.model_name = 'LinearRegression'    

        # Calculate the flattened input size
        flattened_input_size = input_size * sequence_length

        # Linear layer
        self.linear = nn.Linear(flattened_input_size, output_size)

        self.returns = ['prediction']

    def forward(self, x):
        # Flatten the input: (batch_size, sequence_length, input_size) -> (batch_size, input_size * sequence_length)
        x = x.view(x.size(0), -1) 
        return self.linear(x), None
    


class MLPModel(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_layers, dropout):
        super(MLPModel, self).__init__()
        self.model_name = 'MLP'

        # Calculate the flattened input size
        flattened_input_size = input_size * sequence_length

        # Create a list to hold the layers
        layers = []

        # Input layer
        layers.append(nn.Linear(flattened_input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

        self.returns = ['prediction']


    def forward(self, x):
        # Flatten the input: (batch_size, sequence_length, input_size) -> (batch_size, input_size * sequence_length)
        x = x.view(x.size(0), -1) 
        return self.model(x), None


class LSTMModel(nn.Module):
    """
    NN model for time series forecasting.
    """
    #def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_layers, dropout, predict_difference, endogenous_idx):  
        super(LSTMModel, self).__init__()
        self.model_name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.predict_difference = predict_difference
        self.endogenous_idx = endogenous_idx

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout, 
                            )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        #self.attention = nn.Linear(hidden_size, 1) # output size is 1

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        #self.init_weights()

        self.returns = ['prediction', 'final_hidden']


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor of shape (batch_size, input_size)
        """

        # LSTM forward pass
        out, hn = self.lstm(x)  # out: [batch_size, seq_len, hidden_size], hn: [num_layers, batch_size, hidden_size]

        # Only take the output of the last time step
        last_out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout
        last_out = self.dropout(last_out)

        # Pass through fully connected layer
        prediction = self.fc(last_out)  # Shape: (batch_size, input_size)

        if self.predict_difference:
            # If predicting change, add the last value to the prediction
            prediction = x[:, -1, self.endogenous_idx] + prediction

        return prediction, hn[-1] # (the prediction, the final hidden state)





class TransformerModel(nn.Module):
    def __init__(self, 
            input_size, sequence_length, 
            hidden_size, use_embedding_layer,
            output_size, 
            num_heads, num_layers, positional_encoding, dropout, 
            aggregate_hidden,
            predict_difference, endogenous_idx, 
        ):
        super(TransformerModel, self).__init__()
        # https://peterbloem.nl/blog/transformers
        self.model_name = 'Transformer'
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.use_embedding_layer = use_embedding_layer
        self.hidden_size = hidden_size if use_embedding_layer else input_size

        self.output_size = output_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.aggregate_hidden = aggregate_hidden
        self.predict_difference = predict_difference
        self.endogenous_idx = endogenous_idx
                

        # Embedding layer (could represent the hidden size)
        self.embedding = nn.Linear(input_size, hidden_size) if use_embedding_layer else nn.Identity() # in: (batch_size, seq_len, input_size), out: (batch_size, seq_len, hidden_size)


        # Positional encoding to add sequence order information
        self.positional_encoding = PositionalEncoding(hidden_size, dropout, sequence_length) if positional_encoding else None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, # similar to the token embedding size
            nhead=num_heads, # 
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layer to map transformer output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)
        # name to dim
        self.returns = {
            'prediction': output_size,
            'final_hidden': hidden_size
        }


    def forward(self, x):

        # Embed the input if an embedding layer is used
        x = self.embedding(x)

        # Embed the input and add positional encoding
        x = self.positional_encoding(x) if self.positional_encoding else x
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x) # out: [batch_size, seq_len, hidden_size]
        
        # Take the output of the last time step
        final_hidden = x.mean(dim=1) if self.aggregate_hidden else x[:, -1, :]
        
        # Pass through the final projection layer
        prediction = self.fc(final_hidden)

        if self.predict_difference:
            # If predicting change, add the last value to the prediction
            prediction = x[:, -1, self.endogenous_idx] + prediction

        return prediction, final_hidden
    

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / hidden_size))
        pe = torch.zeros(max_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding to input
        return self.dropout(x)



