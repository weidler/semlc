import sys
sys.path.append("../")

import random

import numpy
import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import ConvNet11
from experiment.train import train
from experiment.eval import accuracy

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
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

conv11 = ConvNet11(logdir="transform")

network = conv11

if use_cuda:
    network.cuda()

logger = Logger(network)
logger.save_model(0)

train(net=network,
      num_epoch=200,
      train_set=train_set,
      batch_size=16,
      criterion=nn.CrossEntropyLoss(),
      logger=logger,
      check_loss=100,
      learn_rate=0.001)

network.eval()
print(accuracy(network, test_set, 16))
