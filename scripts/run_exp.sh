#!/usr/bin/env zsh
### run with $ sbatch --job-name=<name> --output=<output> run_exp.sh <filename>
### monitor with $ sacct
### cancel with $ scancel <jobid>

### Job name
#SBATCH --job-name=undefined

### File for the output
#SBATCH --output=undefined.out

### Time your job needs to execute
#SBATCH --time=exp_set_2:00:00

### Memory your job needs per node, e. g. 1G
#SBATCH --mem-per-cpu=2G

### request gpu
#SBATCH --gres=gpu:kepler:1

### run as maastricht university dke project group
#SBATCH --account=um_dke

#SBATCH --mail-type=ALL
#SBATCH --mail-user=julianlehnen@gmail.com

### start environment
source .brains/bin/activate
module load cuda

### Run script
cd ../experiment
python "$1"
