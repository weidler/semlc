#!/bin/bash

#SBATCH --job-name=semlc-gpu
#SBATCH --time=6:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --account=ich020
#SBATCH --constraint=gpu
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load modules
module load daint-gpu
module load cray-python

# load virtual environment
source ${HOME}/lcvenv/bin/activate

# start job
srun python3 -O run_bayes_opt_client.py shallow -i 100 -e 100
