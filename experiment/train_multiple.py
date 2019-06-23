import math
import sys
sys.path.append("../")

import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import BaselineCMap, Baseline, SingleShotInhibitionNetwork
from util.train import train
from util.eval import accuracy

from util.ourlogging import Logger


use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(f"USE CUDA: {use_cuda}.")

# transformation
transform = transforms.Compose([transforms.RandomCrop(24),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

# load data
trainval_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

# Split into train/validation
n_train = math.ceil(0.9 * len(trainval_set))
n_validation = len(trainval_set) - n_train
train_set, validation_set = torch.utils.data.random_split(trainval_set, [n_train, n_validation])

#               0          1      2        3            4               5
strategy = ["baseline", "cmap", "ss", "ss_freeze", "converged", "converged_freeze"][3]
iterations = 10
for i in range(5, iterations):
    logdir = f"{strategy}_{i+1}"
    network = None
    if strategy == "baseline":
        network = Baseline(logdir=logdir)
    elif strategy == "cmap":
        network = BaselineCMap(logdir=logdir)
    elif strategy == "ss":
        network = SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False, logdir=logdir)
    elif strategy == "ss_freeze":
        network = SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True, logdir=logdir)
    elif strategy == "converged":
        raise NotImplementedError("No valid HPOPTIM data.")
    elif strategy == "converged_freeze":
        raise NotImplementedError("No valid HPOPTIM data.")

    print(f"{network.__class__.__name__}_{i+1}")

    if use_cuda:
        network.cuda()

    logger = Logger(network)

    train(net=network,
          num_epoch=160,
          train_set=train_set,
          batch_size=128,
          criterion=nn.CrossEntropyLoss(),
          logger=logger,
          test_set=validation_set,
          learn_rate=0.001,
          verbose=False)

    network.eval()
    logger.log(f"\nFinal Test Accuracy: {accuracy(network, test_set, 128)}")
