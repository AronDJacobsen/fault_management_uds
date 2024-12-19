#!/bin/sh 

### -- set the job Name --
#BSUB -J lstm
### -- specify files --
#BSUB -o /work3/s194262/GitHub/fault_management_uds/hpc_logs/%J-lstm.out
#BSUB -e /work3/s194262/GitHub/fault_management_uds/hpc_logs/%J-lstm.err

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
### -- request _ GB of system-memory --
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"


nvidia-smi
module load cuda/11.8

source /work3/s194262/thesis/bin/activate

cd /work3/s194262/GitHub/fault_management_uds

python fault_management_uds/main.py --config "lstm/design_2.yaml" --num_workers 0


