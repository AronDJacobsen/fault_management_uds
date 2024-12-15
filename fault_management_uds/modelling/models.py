import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from fault_management_uds.utilities import get_accelerator


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
                        target_mean=additional_configurations.get('target_mean', 0.0)
                        )
    
    elif model_name == 'Lag1Predictor':
        model = Lag1Predictor(
                        endogenous_idx=additional_configurations.get('endogenous_idx', [])
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
                        dropout=model_args['dropout']
                        )

    elif model_name == 'Transformer':
        model = TransformerModel(
                        input_size=model_args['input_size'],
                        sequence_length=model_args['sequence_length'], 
                        hidden_size=model_args['hidden_size'], 
                        output_size=model_args['output_size'],
                        num_heads=model_args['num_heads'], 
                        num_layers=model_args['num_layers'], 
                        dropout=model_args['dropout']
                        )

                          
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # use the pre-defined loss function
    criterion = get_loss(training_args['loss_function'])

    # Create a PyTorch Lightning model
    lightning_model = LightningModel(
        model, criterion, 
        learning_rate=training_args['lr'], 
        seed=training_args['seed'],
        skip_optimizer=training_args['skip_optimizer']
        )

    # Load the model state from the checkpoint if a path is provided
    if checkpoint_path is not None:
        lightning_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        
    return lightning_model



class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for training models.
    """

    def __init__(self, model, criterion, learning_rate=0.001, seed=42, skip_optimizer=False):
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.seed = seed
        self.skip_optimizer = skip_optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets, masks, _ = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets, masks)
        self.log('train_loss', loss, prog_bar=True, batch_size=data.size(0), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets, masks, _ = batch
        outputs = self(data)
        loss = self.criterion(outputs, targets, masks)
        self.log('val_loss', loss, prog_bar=True, batch_size=data.size(0))
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def additional_configurations(self, dataset):
        # if the model has additional configurations, call the function
        if hasattr(self.model, 'additional_configurations'):
            self.model.additional_configurations(dataset)
        else:
            print("Model does not have additional configurations.")


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
        return torch.full((batch_size, 1), self.mean.item(), device=x.device)


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
        return x[:, -1, self.endogenous_idx]#.unsqueeze(1)



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

    def forward(self, x):
        # Flatten the input: (batch_size, sequence_length, input_size) -> (batch_size, input_size * sequence_length)
        x = x.view(x.size(0), -1) 
        return self.linear(x)



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

    def forward(self, x):
        # Flatten the input: (batch_size, sequence_length, input_size) -> (batch_size, input_size * sequence_length)
        x = x.view(x.size(0), -1) 
        return self.model(x)



class LSTMModel(nn.Module):
    """
    NN model for time series forecasting.
    """
    #def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.model_name = 'LSTM'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, 
                            )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_size, output_size)

        # Initialize weights
        self.init_weights()


    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)


    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Tensor of shape (batch_size, input_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Only take the output of the last time step
        last_out = out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout
        last_out = self.dropout(last_out)

        # Pass through fully connected layer
        prediction = self.fc(last_out)  # Shape: (batch_size, input_size)

        return prediction

class TransformerModel(nn.Module):
    def __init__(self, input_size, sequence_length, hidden_size, output_size, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.model_name = 'Transformer'

        # Embedding layer for input
        self.input_embedding = nn.Linear(input_size, hidden_size)

        # Positional encoding to add sequence order information
        self.positional_encoding = PositionalEncoding(hidden_size, dropout, max_len=sequence_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,  # Feedforward dimension
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fully connected layer to map transformer output to desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)

        # Project input to hidden size and apply positional encoding
        x = self.input_embedding(x)  # Shape: (batch_size, sequence_length, hidden_size)
        x = self.positional_encoding(x)

        # Permute for transformer: (batch_size, sequence_length, hidden_size) -> (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # Pass through the transformer encoder
        x = self.transformer_encoder(x)  # Shape: (sequence_length, batch_size, hidden_size)

        # Take the output of the last time step
        last_out = x[-1, :, :]  # Shape: (batch_size, hidden_size)

        # Pass through the fully connected layer
        prediction = self.fc(last_out)  # Shape: (batch_size, output_size)

        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register buffer to prevent it from being treated as a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



