#!/bin/sh 

### -- set the job Name --
#BSUB -J anomalous
### -- specify files --
#BSUB -o /work3/s194262/GitHub/fault_management_uds/hpc_logs/anomalous-%J.out
#BSUB -e /work3/s194262/GitHub/fault_management_uds/hpc_logs/anomalous-%J.err

### General options
### –- specify queue --
#BSUB -q gpua100
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 06:00
### -- request _ GB of system-memory --
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"


nvidia-smi
module load cuda/11.8

source /work3/s194262/thesis/bin/activate

cd /work3/s194262/GitHub/fault_management_uds


# default train
python fault_management_uds/train.py --config "linear_regression/5e-4.yaml" --num_workers 0


#------------------------------------------------------------
### -- Iteration 0 --

### Iteration 0
# Training:
#python fault_management_uds/train.py --config "transformer/7_anomalous/0_iteration.yaml" --num_workers 0

# Get features:
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0_250106_0752" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0

# Evaluation:
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0_250106_0752" --data_group "anomalous" --num_workers 0

# Iteration:
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.0_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0_250106_0752" --num_workers 0

#------------------------------------------------------------
### -- Iteration 1 --


### Iteration 0.0
# Train model:
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.0_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0_250106_0752" --num_workers 0

# Get features:
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.0_250107_1459" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0

# Evaluation:
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.0_250107_1459" --data_group "anomalous" --num_workers 0


### Iteration 0.1
# Train model:
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.1_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0_250106_0752" --num_workers 0

# Get features:
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.1_250107_1455" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0

# Evaluation:
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.1_250107_1455" --data_group "anomalous" --num_workers 0



#------------------------------------------------------------
### -- Iteration 2 --


### Iteration 0.0.0
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.0.0_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0.0_250107_1459" --num_workers 0
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.0.0_250108_2052" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.0.0_250108_2052" --data_group "anomalous" --num_workers 0


### Iteration 0.0.1
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.0.1_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0.0_250107_1459" --num_workers 0
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.0.1_250108_2105" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.0.1_250108_2105" --data_group "anomalous" --num_workers 0


### Iteration 0.1.0
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.1.0_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0.1_250107_1455" --num_workers 0
#python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.1.0_250108_2222" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.1.0_250108_2222" --data_group "anomalous" --num_workers 0


### Iteration 0.1.1
#python fault_management_uds/train.py --config "transformer/7_anomalous/0.1.1_iteration.yaml" --fine_tune_path "transformer/7_anomalous/iteration=0.1_250107_1455" --num_workers 0
python fault_management_uds/features.py --model_save_path "transformer/7_anomalous/iteration=0.1.1_250108_2117" --data_types "train" "val" "test" --data_group "anomalous" --num_workers 0
#python fault_management_uds/evaluate.py --model_save_path "transformer/7_anomalous/iteration=0.1.1_250108_2117" --data_group "anomalous" --num_workers 0




