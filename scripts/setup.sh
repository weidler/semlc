#!/bin/bash

# create virtualenv (torch installation fails otherwise)
virtualenv .brains
source .brains/bin/activate

# use cuda
module load cuda

# install pytorch for cuda
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp27-cp27mu-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp27-cp27mu-linux_x86_64.whl
