#!/bin/bash

# create virtualenv (torch installation fails otherwise)
virtualenv --python=/usr/bin/python3.6 .brains
source .brains/bin/activate

# use cuda
module load cuda

# install pytorch for cuda
#torch
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

#torchvision
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

cd ../experiment
python classification.py
