# Biologically Inspired Semantic Lateral Connectivity in Convolutional Neural Networks

### Installation
This module was implemented for Python 3.8, PyTorch 1.7.1 and CUDA 10.1. Required packages can be found in the requirements.txt file.

### Usage
The working directory for all main scripts is `semlc/`. The main script to run experiments is `semlc/run.py`.

Use for instance as follows:
```
cd semlc
python3 run.py alexnet semlc -e 180 -w 3 --data cifar10

# for help, use
python run.py --help
```

### Inspecting Results
Every model is assigned a unique process id. Results and parameters of the model are stored in `semlc/experiments/static/saved_models`. 
We provide a Flask app which can be started from the root directory by calling `./monitor`. 
It provides an interface to navigate and analyze models and their results.

