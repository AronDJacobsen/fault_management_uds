#!/bin/sh 

### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J dummy
### -- specify files --
#BSUB -o /work3/s194262/GitHub/fault_management_uds/hpc_logs/%J-dummy.out
#BSUB -e /work3/s194262/GitHub/fault_management_uds/hpc_logs/%J-dummy.err

### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=shared"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:03
### -- request _ GB of system-memory --
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"


nvidia-smi
module load cuda/11.8

source /work3/s194262/thesis/bin/activate

cd /work3/s194262/GitHub/fault_management_uds

python fault_management_uds/temp.py

