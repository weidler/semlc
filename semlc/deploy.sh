#!/bin/bash

#SBATCH --job-name=semlc
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --account=ich020m
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load modules
module load cray-python

# load virtual environment
source ${HOME}/semlcvenv/bin/activate

# start job
if test -z "$SCRIPTCOMMAND"
then
  python3 -O run.py CLC frozen
else
  eval $SCRIPTCOMMAND
fi