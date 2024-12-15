
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler




class CustomScaler:
    def __init__(self, scaler_type='min-max', feature_range=(0, 1), function_transform_type=None, obvious_min=None, precision=2):

        # store the scaler type
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        self.scaler = self.get_scaler(scaler_type=scaler_type, feature_range=feature_range)

        # store the function transform type
        self.function_transform_type = function_transform_type
        self.function_transformer = FunctionTransformer(transform_type=function_transform_type, obvious_min=obvious_min)

        # store the precision
        self.precision = precision

    def fit(self, data):
        # apply precision
        data = np.round(data, self.precision)
        # apply the transform before fitting the scaler
        data = self.function_transformer.transform(data)
        # fit the scaler
        self.scaler.fit(data)

    def transform(self, data):
        # apply precision
        data = np.round(data, self.precision)
        # apply the transform
        data = self.function_transformer.transform(data)
        # apply the scaler
        data = self.scaler.transform(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        # apply the inverse scaler
        data = self.scaler.inverse_transform(data)
        # apply the inverse transform
        data = self.function_transformer.inverse_transform(data)
        # apply precision
        data = np.round(data, self.precision)

        return data

    ### Scaler functions
    def get_scaler(self, scaler_type='min-max', feature_range=(0, 1)):
        if scaler_type == 'min-max':
            return MinMaxScaler(feature_range=feature_range)
        elif scaler_type == 'standard':
            return StandardScaler()
        else:
            raise ValueError(f"Invalid scaler_type {scaler_type}.")



class FunctionTransformer:

    def __init__(self, transform_type=None, obvious_min=None):
        self.transform_type = transform_type
        
        # store the minimum value (used to set 0 as the minimum value)
        # if obvious_min is None or obvious_min > 0:
        #     print("HERE: ", obvious_min)
        #     self.shift = 0
        # else:
        self.shift = obvious_min

        self.transform, self.inverse_transform = self.get_transform(transform_type)

    def get_transform(self, transform_type=None):
        if transform_type == 'log':
            return self.log_transform, self.inverse_log_transform
        elif transform_type == 'sqrt':
            return self.sqrt_transform, self.inverse_sqrt_transform
        # If none, then return identity transform
        elif transform_type is None or transform_type == 'identity' or transform_type.lower() == 'none':
            return self.identity_transform, self.inverse_identity_transform
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    # Log transform
    def log_transform(self, data):
        # shift the data to ensure no negative values
        return np.log1p(data - self.shift)
    
    def inverse_log_transform(self, data):
        # shift the data to ensure no negative values
        return np.expm1(data) + self.shift

    # Square root transform
    def sqrt_transform(self, data):
        return np.sqrt(data - self.shift + 1e-6)  

    def inverse_sqrt_transform(self, data):
        return np.square(data) + self.shift - 1e-6

    # Identity transform
    def identity_transform(self, data):
        return data

    def inverse_identity_transform(self, data):
        return data
