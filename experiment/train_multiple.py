import math
import sys

from torch.optim import SGD

from model.network.VGG import vgg19, vgg19_inhib

sys.path.append("../")

import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import BaselineCMap, Baseline, SingleShotInhibitionNetwork, ConvergedInhibitionNetwork
from util.train import train
from util.eval import accuracy

from util.ourlogging import Logger


use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(f"USE CUDA: {use_cuda}.")

# transformation
transform = transforms.Compose([transforms.RandomCrop(32, 4),
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

#               0          1      2        3            4               5               6           7               8
strategy = ["baseline", "cmap", "ss", "ss_freeze", "converged", "converged_freeze", "vgg19", "vgg19_inhib", "vgg19_inhib_self"][8]
iterations = 10
for i in range(0, iterations):
    logdir = f"{strategy}_{i+1}"
    network = None
    if strategy == "baseline":
        network = Baseline()
    elif strategy == "cmap":
        network = BaselineCMap()
    elif strategy == "ss":
        network = SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False)
    elif strategy == "ss_freeze":
        network = SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True)
    elif strategy == "converged":
        network = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False)
    elif strategy == "converged_freeze":
        network = ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True)  # toeplitz
    elif strategy == "vgg19":
        network = vgg19()
    elif strategy == "vgg19_inhib":
        network = vgg19_inhib()
    elif strategy == "vgg19_inhib_self":
        network = vgg19_inhib(self_connection=True)

    print(f"{network.__class__.__name__}_{i+1}")

    if use_cuda:
        network.cuda()

    logger = Logger(network, experiment_code=f"{strategy}_{i}")

    train(net=network,
          num_epoch=100,
          train_set=train_set,
          batch_size=128,
          criterion=nn.CrossEntropyLoss(),
          logger=logger,
          val_set=validation_set,
          optimizer=SGD(network.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
          learn_rate=0.05,
          verbose=False)

    network.eval()
    logger.log(f"\nFinal Test Accuracy: {accuracy(network, test_set, 128)}")
