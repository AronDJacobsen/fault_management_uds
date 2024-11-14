

import numpy as np
import pandas as pd
import random

class AnomalyInjector:
    def __init__(self, time_series):
        self.time_series = time_series
        self.series_mean = np.mean(time_series)
        self.series_std = np.std(time_series)
        
    def generate_duration(self, duration):
        """Generates a duration based on given configuration (fixed, range, or normal distribution)."""
        if isinstance(duration, int):
            return duration
        elif isinstance(duration, list) and len(duration) == 2:
            return random.randint(*duration)
        elif isinstance(duration, dict) and 'mean' in duration and 'std' in duration:
            return max(1, int(np.random.normal(duration['mean'], duration['std'])))
        else:
            raise ValueError("Invalid duration format.")

    def generate_intensity(self, intensity):
        """Generates intensity based on configuration (fixed, range, or normal distribution)."""
        if isinstance(intensity, (int, float)):
            return intensity * self.series_std
        elif isinstance(intensity, list) and len(intensity) == 2:
            return random.uniform(*intensity) * self.series_std
        elif isinstance(intensity, dict) and 'mean' in intensity and 'std' in intensity:
            return np.random.normal(intensity['mean'], intensity['std']) * self.series_std
        else:
            raise ValueError("Invalid intensity format.")

    def inject_frozen(self, start_time, duration):
        """Freezes the time series values at a constant value over the specified duration."""
        end_time = start_time + self.generate_duration(duration)
        value = self.time_series[start_time]
        self.time_series[start_time:end_time] = value
        return self.time_series

    def inject_spike(self, start_time, duration, intensity):
        """Injects a spike with specified intensity over the given duration."""
        end_time = start_time + self.generate_duration(duration)
        intensity_value = self.generate_intensity(intensity)
        self.time_series[start_time:end_time] += intensity_value
        return self.time_series

    def inject_pulse(self, start_time, duration, intensity):
        """Injects a pulse, alternating between spike and normal value for the duration."""
        end_time = start_time + self.generate_duration(duration)
        pulse_intensity = self.generate_intensity(intensity)
        for i in range(start_time, end_time, 2):
            self.time_series[i] += pulse_intensity
        return self.time_series

    def inject_noise(self, start_time, duration, intensity):
        """Injects random noise over the duration, based on intensity."""
        end_time = start_time + self.generate_duration(duration)
        noise_intensity = self.generate_intensity(intensity)
        noise = np.random.normal(0, noise_intensity, end_time - start_time)
        self.time_series[start_time:end_time] += noise
        return self.time_series

    def inject_offset(self, start_time, duration, intensity):
        """Applies a constant offset to the time series values over the duration."""
        end_time = start_time + self.generate_duration(duration)
        offset_value = self.generate_intensity(intensity)
        self.time_series[start_time:end_time] += offset_value
        return self.time_series

    def inject_drift(self, start_time, duration, intensity):
        """Gradually increases values over the duration (linear drift)."""
        end_time = start_time + self.generate_duration(duration)
        drift_amount = self.generate_intensity(intensity)
        drift_values = np.linspace(0, drift_amount, end_time - start_time)
        self.time_series[start_time:end_time] += drift_values
        return self.time_series

    def inject_anomalies(self, start_time, anomalies):
        """Injects multiple anomalies at a specified start time based on a config dictionary."""
        for anomaly in anomalies:
            anomaly_type = anomaly['type']
            duration = anomaly.get('duration', 5)
            intensity = anomaly.get('intensity', 1.0)
            
            if anomaly_type == 'frozen':
                self.inject_frozen(start_time, duration)
            elif anomaly_type == 'spike':
                self.inject_spike(start_time, duration, intensity)
            elif anomaly_type == 'pulse':
                self.inject_pulse(start_time, duration, intensity)
            elif anomaly_type == 'noise':
                self.inject_noise(start_time, duration, intensity)
            elif anomaly_type == 'offset':
                self.inject_offset(start_time, duration, intensity)
            elif anomaly_type == 'drift':
                self.inject_drift(start_time, duration, intensity)
            else:
                raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        return self.time_series


# Example usage
time_series = pd.Series(np.sin(np.linspace(0, 20, 1000)))  # Sample time series

# Create the injector
injector = AnomalyInjector(time_series)

# Define anomalies
anomalies = [
    {"type": "spike", "duration": [5, 10], "intensity": {"mean": 2, "std": 0.5}},
    {"type": "frozen", "duration": 4},
    {"type": "drift", "duration": {"mean": 5, "std": 2}, "intensity": [0.1, 0.5]},
]

# Inject anomalies starting at index 100
start_time = 100
modified_series = injector.inject_anomalies(start_time, anomalies)

# Plot the original and modified time series for comparison (optional)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(time_series, label="Original Time Series")
plt.plot(modified_series, label="Modified Time Series with Anomalies", linestyle='--')
plt.legend()
plt.show()
