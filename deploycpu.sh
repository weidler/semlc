#!/bin/bash -l

#SBATCH --job-name=minus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=admin@tonioweidler.de
#SBATCH --time=12:00:00
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
source ${HOME}/lcvenv/bin/activate

# start job
python3 -O main.py CLC frozen