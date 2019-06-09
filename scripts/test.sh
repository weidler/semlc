#!/usr/bin/env zsh
### DO NOT RUN YET, WORK IN PROGRESS

### Job name
#SBATCH --job-name=TESTJOB

### File for the output
#SBATCH --output=TESTJOB_OUTPUT

### Time your job needs to execute, e. g. 0 min 10 sec
#SBATCH --time=00:00:10

### Memory your job needs per node, e. g. 1G
#SBATCH --mem=2M

### request gpu
#SBATCH --gres=gpu:pascal:1

### run as maastricht university dke project group
#SBATCH --account=um_dke

### Change to working directory
cd ../experiment

python classification.py
