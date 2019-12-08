#!/usr/bin/env zsh
### run with $ sbatch test.sh
### monitor with $ sacct
### cancel with $ scancel <jobid>

### Job name
#SBATCH --job-name=ALEXNET

### File for the output
#SBATCH --output=ALEXNET_OUTPUT

### Time your job needs to execute
#SBATCH --time=exp_set_2:00:00

### Memory your job needs per node, e. g. 1G
#SBATCH --mem-per-cpu=2G

### request gpu
#SBATCH --gres=gpu:pascal:1

### run as maastricht university dke project group
#SBATCH --account=um_dke

### Run script
./setup.sh


