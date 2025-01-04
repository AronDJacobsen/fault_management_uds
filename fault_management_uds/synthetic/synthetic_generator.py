
import numpy as np
import random
from itertools import groupby



class AnomalyGenerator:
    # no need for init
    def inject_anomaly(self, time_series, value_col_idx, anomaly_type, start_idx, end_idx, duration, magnitude):
        """Inject a specific type of anomaly into the time series."""

        # Inject the specified anomaly type
        if anomaly_type == "spike":
            time_series = self.inject_spike(time_series, value_col_idx, start_idx, end_idx, duration, magnitude)
        elif anomaly_type == "noise":
            time_series = self.inject_noise(time_series, value_col_idx, start_idx, end_idx, duration, magnitude)
        elif anomaly_type == "frozen":
            time_series = self.inject_frozen(time_series, value_col_idx, start_idx, end_idx, duration, magnitude)
        elif anomaly_type == "offset":
            time_series = self.inject_offset(time_series, value_col_idx, start_idx, end_idx, duration, magnitude)
        elif anomaly_type == "drift":
            time_series = self.inject_drift(time_series, value_col_idx, start_idx, end_idx, duration, magnitude)
        
        return time_series

    def inject_spike(self, time_series, value_col_idx, start_idx, end_idx, duration, magnitude):
        """Inject a single spike anomaly."""

        midpoint = duration // 2
        # controls 2 things: 1) uneven, increase linspace length, 2) uneven, do not include midpoint in the last half
        even_uneven_parameter = duration % 2 # 0 if even, 1 if uneven
        first_half = np.linspace(0, magnitude, midpoint+1+even_uneven_parameter)[1:] # don't include the first 0
        # if even, include midpoint in the last half
        # if uneven, do not include midpoint in the last half
        last_half = first_half[::-1][even_uneven_parameter:].copy()
        pattern = np.concatenate([first_half, last_half])

        time_series.iloc[start_idx:end_idx, value_col_idx] += pattern 
        return time_series

    def inject_noise(self, time_series, value_col_idx, start_idx, end_idx, duration, magnitude):
        """Inject random noise anomaly over a duration."""
        noise = np.random.normal(0, magnitude, duration)
        time_series.iloc[start_idx:end_idx, value_col_idx] += noise
        return time_series

    def inject_frozen(self, time_series, value_col_idx, start_idx, end_idx, duration, magnitude):
        """Inject a frozen anomaly (constant value)."""
        frozen_value = time_series.iloc[start_idx, value_col_idx]
        time_series.iloc[start_idx:end_idx, value_col_idx] = frozen_value
        return time_series

    def inject_offset(self, time_series, value_col_idx, start_idx, end_idx, duration, magnitude):
        """Inject an offset anomaly (shifted baseline over duration)."""
        time_series.iloc[start_idx:end_idx, value_col_idx] += magnitude
        return time_series

    def inject_drift(self, time_series, value_col_idx, start_idx, end_idx, duration, magnitude):
        """Inject a drift anomaly (gradual increase over duration)."""
        drift = np.linspace(0, magnitude, end_idx - start_idx)
        time_series.iloc[start_idx:end_idx, value_col_idx] += drift
        return time_series



def find_unterrupted_sequences(lst, end_buffer):
    unterrupted_sequences = []
    unterrupted_sequences_sets = [] # with (start, end)
    # Group consecutive numbers by their difference
    for k, g in groupby(enumerate(lst), lambda x: x[0] - x[1]): # x[0] is the index, x[1] is the value, for gaps the difference increases
        # Extract the start and end of each group, k is the difference, g is the group i.e. (index, value)
        group = list(map(lambda x: x[1], g)) # x[1] is the value, g is the group/        
        if len(group) > end_buffer:
            start, end = group[0], group[-1] - end_buffer
            unterrupted_sequences.extend(list(range(start, end+1)))
            unterrupted_sequences_sets.append((start, end))
    return unterrupted_sequences, unterrupted_sequences_sets



class AnomalyHandler:
    def __init__(self, anomaly_config, anomaly, value_col, 
                 n_obs, magnitude_scale, obvious_min, obvious_max, 
                 seed=42, buffer=12*60, edge_buffer=7*24*60, available_indices=None):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary for the anomalies.
        time_series : pd.Series
            Time series data to inject anomalies into.
        anomaly : str
            Type of anomaly to inject.
        magnitude_scale : list
            Range of magnitudes to inject.
        seed : int
            Random seed for reproducibility.
        """
        # define
        self.n_obs = n_obs
        self.anomaly = anomaly
        self.seed = seed
        self.value_col = value_col if isinstance(value_col, str) else value_col[0]


        # The minimum gap between anomalies
        self.buffer = buffer
        # Avoiding the first and last _of the time series
        self.edge_buffer = edge_buffer

        self.available_indices = available_indices

        # set configuration parameters
        self.anomaly_config = anomaly_config
        self.proportion = self.anomaly_config['proportion']
        # set the magnitude range based on e.g. min-max or std
        # - unique for each sensor
        self.magnitude_scale = magnitude_scale
        self.obvious_min = obvious_min
        self.obvious_max = obvious_max

        # define duration parameters
        self.duration_mean = self.anomaly_config['duration']['normal']['mean']
        self.duration_std = self.anomaly_config['duration']['normal']['std']
        # define magnitude parameters
        self.magnitude_min = self.anomaly_config['magnitude']['range']['min']
        self.magnitude_max = self.anomaly_config['magnitude']['range']['max']
        self.magnitude_sign = self.anomaly_config['magnitude']['sign']
        
        # initialize the injections
        self.total_injections = 0
        self.start_indices = []
        self.durations = []
        self.magnitudes = []

        # store the anomaly generator
        self.anomaly_generator = AnomalyGenerator()


    def initialize_injections(self):
        """
        Initialize the injections based on the configuration.
        - Number of injections are based on the proportion defined and the sampled durations.
        """

        # # set the seed to reset and ensure reproducibility
        # np.random.seed(self.seed)

        # find the total duration of the anomalies based on the proportion
        self.total_duration = round(self.proportion * self.n_obs)

        # get the durations until the total duration is reached
        self.durations = []
        current_duration_sum = 0
        while sum(self.durations) < self.total_duration:
            # Generate samples
            duration_sample = self.sample_duration()
            self.durations.append(duration_sample)
            current_duration_sum += duration_sample
        
        # total injections
        self.total_injections = len(self.durations)
        print(f"Total injections: {self.total_injections}")
        
        # sample the magnitudes
        self.magnitudes = self.sample_magnitudes()

        # get start indices
        self.start_indices = self.get_start_indices()

        # TODO: evaluate consistency?


    def sample_duration(self):
        """Sample a duration from a normal distribution."""
        # Generate samples
        sample = np.random.normal(loc=self.duration_mean, scale=self.duration_std, size=1)[0]
        # Round to integer and clip the duration to minimum 1 minute
        sample = np.clip(int(np.round(sample)), 1, None)
        return sample


    def sample_magnitudes(self):
        """
        Sample magnitudes from a specified range distribution.
        - handle None, rounding, direction
        """
        # If None, for e.g. Frozen
        if self.magnitude_min is None or self.magnitude_max is None:
            # return an array of 1 (identity), use total_injections
            return np.ones(self.total_injections)
        
        # sample the magnitudes
        samples = np.random.uniform(self.magnitude_min, self.magnitude_max, self.total_injections)
        
        # round to 3 decimals
        samples = np.round(samples, 3)
        
        # handle directions
        if self.magnitude_sign == '+':
            samples = np.abs(samples)
        elif self.magnitude_sign == '-':
            samples = -np.abs(samples)
        elif self.magnitude_sign == '+-':
            # sample from both sides
            pos_neg = np.random.choice([-1, 1], self.total_injections)
            samples = samples * pos_neg
        
        # Apply the scale
        samples = samples * self.magnitude_scale
        return samples


    def get_start_indices(self):
        """
        Based on total injections and selected durations, place the start indices.
        
        Note: 
        - should be placed somewhat evenly around
        - not too close to the start or end
        - not too close to each other
        """

        max_duration = max(self.durations)
        
        # Available indices to place anomalies        
        if self.available_indices is not None:
            # User defined available indices, adjust to handle interruptions
            available_indices, _ = find_unterrupted_sequences(self.available_indices, max_duration)
            # handle edge wrt max duration
            available_indices = available_indices[:-max_duration]

        else:
            available_indices = list(range(self.edge_buffer, self.n_obs - self.edge_buffer))

        
        # handle cases if the edge buffer is in the available indices
        if self.edge_buffer-1 in available_indices:
            print("Edge buffer is in available indices.")
            # remove indices up to the edge buffer
            edge_idx = available_indices.index(self.edge_buffer-1)
            available_indices = available_indices[edge_idx+1:]
        # same for the last index
        if self.n_obs - self.edge_buffer in available_indices:
            print("Last index is in available indices.")
            # remove indices up to the edge buffer
            edge_idx = available_indices.index(self.n_obs - self.edge_buffer)
            available_indices = available_indices[:edge_idx]




        if not available_indices:
            print("No available indices to place anomalies.")
            return []

        # Randomly pick start indices for the anomalies
        start_indices = []

        for i in range(int(self.total_injections)):
            # Check if there no valid choices left
            if not available_indices:
                print(f"No valid choices, breaking early: injected {i} out of intented {self.total_injections}")
                break

            # Randomly select an index
            avail_start_idx = random.randint(0, len(available_indices))
            start_idx = available_indices[avail_start_idx]
            duration = self.durations[i]

            # Add the index to the list of start indices
            start_indices.append(start_idx)

            # Update the list of available indices
            del available_indices[avail_start_idx - max_duration - self.buffer:avail_start_idx + duration + self.buffer]
            

        # In case of early break, update:
        self.total_injections = len(start_indices)
        self.durations = [self.durations[i] for i in range(self.total_injections)]
        self.magnitudes = [self.magnitudes[i] for i in range(self.total_injections)]
        
        return start_indices


    def evaluate_consistency(self, action='remove'):
        """Evaluate the consistency of the injections."""
        # sort the start indices
        self.start_indices = sorted(self.start_indices)
        i = 1
        while i < self.total_injections:
            # check if the current start index is less than the previous end index
            if self.start_indices[i] < self.start_indices[i-1] + self.durations[i-1]:
                # overlap detected
                print(f"Overlap detected: {self.start_indices[i-1]}-{self.start_indices[i-1] + self.durations[i-1]} and {self.start_indices[i]}-{self.start_indices[i] + self.durations[i]}")
                if action == 'remove':
                    # remove the current injection
                    del self.start_indices[i]
                    del self.durations[i]
                    del self.magnitudes[i]
                    self.total_injections -= 1
                    # do not increment i, as we need to check the new current element
                else:
                    i += 1
            else:
                i += 1

    def inject_anomalies(self, time_series):
        """Inject anomalies into the time series based on the configuration."""
        # make a copy of the time series
        self.time_series = time_series.copy()
        value_col_idx = self.time_series.columns.get_loc(self.value_col)
        # iterate over the anomalies
        for i in range(self.total_injections):
            start_idx = self.start_indices[i]
            duration = self.durations[i]
            magnitude = self.magnitudes[i]
            end_idx = start_idx + duration
            # inject the anomaly
            self.time_series = self.anomaly_generator.inject_anomaly(self.time_series, value_col_idx, self.anomaly, start_idx, end_idx, duration, magnitude)
        # cap values exceeding obvious min and max
        self.time_series[self.value_col] = np.clip(self.time_series[self.value_col], self.obvious_min, self.obvious_max)
        
        return self.time_series


    # Helper functions
    def get_indicator(self):
        """Return a 0-1 mask for the injected anomalies."""
        indicator = np.zeros(self.n_obs)
        for i in range(self.total_injections):
            start_time = self.start_indices[i]
            duration = self.durations[i]
            indicator[start_time:start_time + duration] = 1
        return indicator
    
    def set_injection_start(self, start_idx):
        # set an injections based on a start index
        self.start_indices.append(start_idx)
        self.total_injections += 1
        self.durations.append(self.sample_duration())
        self.magnitudes.append(self.sample_magnitudes()[0])

        # adjust in case it exceeds the time series length
        if start_idx + self.durations[-1] >= self.n_obs:
            # adjust the duration
            self.durations[-1] = self.n_obs - start_idx
            # if drift, also adjust the magnitude
            if self.anomaly in ['drift']:
                # adjust the magnitude relative to the duration (so it still follow the same slope)
                self.magnitudes[-1] = self.magnitudes[-1] * (self.durations[-1] / self.durations[-1])

        # evaluate the consistency to ensure no overlap
        self.evaluate_consistency()

