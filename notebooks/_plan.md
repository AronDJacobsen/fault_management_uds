

## 1. Pilot:

*Model*: Simple NN with 1 hidden layer, 32 units, 0.001 learning rate

*Data*: Sensor `G72F040` due to fast response time to rainfall.


**Experiments**

- Scaler: MinMaxScaler, StandardScaler

- Data transformation: None, Log, Sqrt

- Feature engineering: Time of day, Day of week, Both?

- Data augmentation: None, Noise


**Excluded**

- *Optimization*: simple and quick modelling, no need for comprehensive optimization
- *Lagged time*: goal is multi-variate forecasting; lagging for one variable would require lagging for all variables.


# 2. Benchmarks

**Baseline Models**:
- Mean predictor
- Previous value predictor
- Linear regression
- XGBoost
- Neural network


**Advanced Models**:

- LSTM
- Transformer
- ST-GNN


## 4. ?Experiment further with best?:




## Forecasting

- Evaluate steps ahead prediction.
- Evaluate different weather conditions.


## Anomaly Management

#### Detection

*Metrics*: Precision, Recall, Timing.


#### Diagnosis

*Known*: Technical fault.


*Provide*: Visualization tools.

*Possible*:
- Classify faults based on data characteristics.
- Extend synthetic to also generate system-wide faults.


#### Correction

Evaluate **imputation** performance of the correctly identified anomalies based on the **MAE deviation** from the original data.


*Possible*:
- Iterative error removal


## Sensor expansion

- How effective at adding a new sensor to the system?


## Surrogate modelling

- How effective at predicting the system behaviour?
- Comparison with the Mike simulation model.


## Provide a cleaned dataset?




