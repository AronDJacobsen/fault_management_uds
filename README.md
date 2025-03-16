# Multi-Feature Anomaly Detection in Urban Drainage Systems

#### *using Transformers, Isolation Forest & Targeted Iterative Refinement*

##
## Introduction

This repository contains the code my thesis on modelling and anomaly detection in urban drainage systems.

In general, the project has contributed to:

- Improved Data Processing of the Bellinge Dataset
- Analysis of the Bellinge Dataset
- Modelling of the Bellinge Dataset
- Anomaly Detection using synthetic anomalies


### Using the Repository



These are the main scripts in the project in a chronological order (see `submit_gpu.sh` for use-cases):

*Data Processing*:

- `notebooks/0_processing/1_processing_data.ipynb`
- `notebooks/0_processing/2_handling_data.ipynb`

*Modelling*:

- `fault_management_uds/train.py`
  - See the `experiments` folder for the different experiments
  - See `notebooks/3_evaluation/` for further evaluation of the models

*Anomaly Detection*:

- `notebooks/1_synthetic/create_polluted_data.ipynb`
- Then you can run these scripts in the `fault_management_uds/` folder:
  1. `train.py` for training on the polluted data.
  2. `features.py` for extracting features for anomaly detection.
  3. `evaluate.py` for evaluating the anomaly detection.


##
**Due to time contraints**, the documentation could be improved. However, you are always welcome to contact me for further information or help.


#
#
---

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modelling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fault_management_uds and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fault_management_uds   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fault_management_uds a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modelling
    │
    ├── modelling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Python Environment


Freeze the environment to a file:

    ```bash
    pip freeze | grep -v "file://" > requirements.txt
    ```
    or windows
    ```bach
    pip freeze | findstr /V "file://" > requirements.txt
    ```

    ```bash
    pip install -r requirements.txt
    ```

And, install repo as editable:

    ```bash
    pip install -e .
    ```

Activate:

    ```bash
    source /work3/s194262/thesis/bin/activate
    ```


## HPC

Submit job:

    ```bash
    bsub < submit.sh
    ```

Kill job:

    ```bash
    bkill -J <job_id>
    ```

Check job:

    ```bash
    bjobs
    ```



Links for HPC:
1. [SSH and ThinLinc login using ssh-keys](https://www.hpc.dtu.dk/?page_id=4317)
2. [Moving files to and from the cluster using ssh keys](https://www.hpc.dtu.dk/?page_id=4377#filezilla)
- Proxy should be `None` in FileZilla


