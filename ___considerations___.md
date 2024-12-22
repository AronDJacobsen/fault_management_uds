
# *Anomaly management*: considerations

This document is a part of the *Anomaly management* project. It contains the considerations and difficulties that are encountered and addressed during the project.



---
# Dump

#### TOOD:

- more flexible handling of best model path -> make it relative path!



#### Cleaning the whole dataset

- train a AE on the whole dataset
- apply KMeans to the latent space, elbow method to find the optimal number of clusters
- assign each data point to a cluster
- then split each cluster into train, val and test %
  - but sorted by time
  - this could ensure both learns and tests on all the data distribution


####  Anomaly detection

- Use Isolation Forest to detect anomalies
- Build on data to create a vector; then test improvements
  - first on prediction error
    - then t-1 and t+1 errors also
  - then with the target (t, but also t-1 and t+1)
  - cosine similarity between
    - attention weights at t and t+1
    - hidden states at t and t+1
  - pre-whitening of the data and residuals? or seasonal decomposition?
    - maybe not needed with target is included
  - hmm temporal also? the sin and cos?
  

#### Locally Integrated Gradients

- After an outlier has been identified
- Consider the time step where the outlier is the input values
  - The baseline will be what the model predicted the outlier value to be
  - Then, for e.g. each 10% intensity/interpolation, calculate the integrated gradients
    - and store the value for the time step as a vector -> dim=10
  - Then, can you use this for something?

#### Other

- is this close to the concept of kalman filtering?
- flow: input data, attention, hidden state, output, target, error, input contribution


---

# Considerations

Simplifications
-------------------------------------

  - [x] Rainfall and single sensor data



Pre-requisites
-------------------------------------

  - [x] Proper data handling
  - [ ] Robust modelling
  - [x] Evaluation framework



Difficulties
-------------------------------------

  - [ ] Identifying the start and end of an anomaly
  - [ ] Differentiating between anomalies and rare events

---

# Solutions

### Proper data handling

  - Data cleaning
  - Data handling
  - Preparation of the data to improve modelling


### Robust modelling

Baseline models
  - Mean prediction
  - Previous value prediction
  - Linear regression
  - XGBoost

Advanced models
  - Neural networks
  - LSTM
  - Transformer
  - ?Temporal Fusion Transformer?


### Evaluation framework


Synthetic error generation

Outlier detection
  - Prediction error: model output
  - Attention weights: model data usage
  - Integrated Gradients: model input

Metrics


### Identifying anomalous periods

**First**, investigate the outlier detection methods that can be used to identify anomalies.

Based on the prediction error:
- Threshold-based: basic 3-sigma rule
- Condition-aware thresholding: 3-sigma rule based dry and not dry conditions
- Model-based: using an ML model (Isolation Forest) to identify anomalies



### Differentiating between anomalies and rare events

The above?


Integrated Gradients
- Based on low integrated gradients
  1. If the model has been trained on anomalies
  2. It learns to ignore them and focus on the normal data
  3. Which means that the integrated gradients will be low for anomalies
  4. Rare events should have high integrated gradients since it utilizes their information


---

## Four Data Scenarios


  1. **Normal Data**
  2. **Rare but Valid Data**
  3. **Seen Anomalous Data**
  4. **Unseen Anomalous Data**


### Factors to Consider with Respect to the Four Data Scenarios


#### 1. **Residuals (Prediction Error)**
The difference between the model's predicted values and the actual observed values.

- **Normal Data:** should be small and random, since the model has learned the underlying patterns.
- **Rare but Valid Data:** might increase due to the model not fully understanding these rare instances.
- **Seen Anomalous Data:** likely increase, as the model recognizes these anomalies, but may be affected by the full duration.
- **Unseen Anomalous Data:** likely be high as the model has no context for this data.


#### 2. **Attention (Within Model)**
Attention mechanisms allow the model to focus on the most important parts of the data.

- **Normal Data:** should focus on the regular patterns in the data, common trends and relationships within the features.
- **Rare but Valid Data:** should attend this data, since it is based on the system behavior and data distribution.
- **Seen Anomalous Data:** likely not attend them, or it will? because either the model has learned to ignore them or to focus on them.
- **Unseen Anomalous Data:** likely misinterpret and fail to properly attend to the data.





#### 3. **Contribution (Output)**
The contribution factor indicates which input features are most influential in determining the model’s predictions.

- **Normal Data:** The contribution of input features to predictions will likely follow expected patterns.
- **Rare but Valid Data:** The contributions may vary slightly from the normal, but should still be within a reasonable range.
- **Seen Anomalous Data:** attend or ignore?
- **Unseen Anomalous Data:** The contribution of input features could become unpredictable and erratic.


### ALSO:

Consider that the data is probably highly auto-correlated, meaning that the model will likely focus on the most recent data points.


What about the iterative approach???

What about dynamic environments causing a higher error rate?
- input rain/external input into isolation forest?
- ARIMA to predict the residuals given input? then adapt to environments with high error rates? and adjust? similar to pre-withning?
- increasing intensity of injected error, mae as a function of intensity, is there e.g. a elbow point where the error rate increases significantly? or visually
- 
