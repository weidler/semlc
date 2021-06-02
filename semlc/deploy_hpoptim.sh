#!/bin/bash

#SBATCH --job-name=semlc-gpu
#SBATCH --time=24:00:00
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
if test -z "$SCRIPTCOMMAND"
then
  srun python3 -O run.py CLC frozen
else
  eval $SCRIPTCOMMAND
fi