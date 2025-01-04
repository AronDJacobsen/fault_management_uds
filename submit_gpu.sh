#!/bin/sh 

### -- set the job Name --
#BSUB -J anomalous
### -- specify files --
#BSUB -o /work3/s194262/GitHub/fault_management_uds/hpc_logs/anomalous-%J.out
#BSUB -e /work3/s194262/GitHub/fault_management_uds/hpc_logs/anomalous-%J.err

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 03:00
### -- request _ GB of system-memory --
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"


nvidia-smi
module load cuda/11.8

source /work3/s194262/thesis/bin/activate

cd /work3/s194262/GitHub/fault_management_uds

# Training:
#python fault_management_uds/train.py --config "transformer/7_anomalous/hold_out_endo.yaml" --num_workers 0

# Evaluation:
python fault_management_uds/get_outputs.py --model_save_path "transformer/7_anomalous/name=1_iteration_250101_2145" --data_types ["test"] --data_group "anomalous" --num_workers 0

# Iteration:
#python fault_management_uds/train.py --config "transformer/7_anomalous/hold_out_endo.yaml" --num_workers 0

