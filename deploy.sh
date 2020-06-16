#!/bin/bash

#SBATCH --job-name=minus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=admin@tonioweidler.de
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# load modules
module load daint-gpu
module load cray-python
module load cudatoolkit/10.0.130_3.22-7.0.1.0_5.2__gdfb4ce5

# load virtual environment
source ${HOME}/lcvenv/bin/activate

python3 -uO main.py CLC frozen