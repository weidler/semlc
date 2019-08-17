import sys
sys.path.append("../")

import random

import numpy
import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import InhibitionNetwork
from util.train import train
from util.eval import accuracy

from util.ourlogging import Logger

torch.random.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

transform = transforms.Compose([transforms.RandomCrop(24),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

strategy = "toeplitz"
scope = 27
ricker_width = 3
damp = 0.1
net = InhibitionNetwork(logdir=f"{strategy}/scope_{scope}/width_{ricker_width}/damp_{damp}",
                           scope=[scope],
                           width=ricker_width,
                           damp=0.1,
                           inhibition_depth=1,
                           inhibition_strategy=strategy,
                           )

network = net

if use_cuda:
    network.cuda()

logger = Logger(network)

train(net=network,
      num_epoch=180,
      train_set=train_set,
      batch_size=128,
      criterion=nn.CrossEntropyLoss(),
      logger=logger,
      test_set=test_set,
      learn_rate=0.001)

network.eval()
print(accuracy(network, test_set, 128))
