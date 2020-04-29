# Biologically Inspired Lateral Connectivity in Convolutional Neural Networks

## User Guide

### Installation

This module was implemented for Python 3.6.8. All required packages can be found in the requirements.txt file.

When using CUDA the following PyTorch and torchvision versions are needed:
```bash
# torch
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl

# torchvision
$ pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

### Usage
The working directory for all main scripts is the root folder `inhibition-net/` of this repository.
The main script used for experiment is `experiments/train_multiple.py` which requires the strategy as argument.
Optionally, the number of iterations can be passed as an argument.

```
python experiment/train_multiple.py baseline

# use for help, all available strategies and additional optional parameters
python experiment/train_multiple.py --help

```

A different script for CapsNet is available as it uses a different train function than our main experiments.
```
python model/network/capsnet.py {capsnet, inhib_capsnet} 

# use for help and additional optional parameters
python model/network/capsnet.py --help

```
### Output
Every Model generates a unique process id used in the file names for saved models and optimizers, log files and a keychain.txt to lookup the belonging experiment configurations.
The keychain contains tab-separated the id, the experiment group and an iteration index (i.e. baseline_15), the representation of the model and a timestamp.
This keychain is used to load saved models for visualizations and analysis.

Please note that we renamed our strategies for the paper submission but left the code as is.
Find a mapping table below for the codes and class names we used.

| Strategy       | Code             | Name of class            |
|----------------|------------------|-----------------------------|
| None           | baseline         | Baseline                    |
| LRN            | cmap             | BaselineCMap                |
| SSLC Frozen    | ss_freeze        | SingleShotInhibitionNetwork |
| SSLC Adaptive  | ss               | SingleShotInhibitionNetwork |
| CLC Frozen     | converged_freeze | ConvergedInhibitionNetwork  |
| CLC Adaptive   | converged        | ConvergedInhibitionNetwork  |
| CLC Parametric | parametric       | ParametricInhibitionNetwork | 

### Testing
Using ``python experiments/network_accuracy.py`` the test accuracy with confidence interval for a number of specified strategies can be computed.

Different visualization and analysis scripts are available - see python documentation strings for usage.

