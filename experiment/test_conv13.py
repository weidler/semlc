import time
import numpy
import random
import sys
sys.path.append("../")

import torch
import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import Baseline
from util.eval import accuracy
from util.ourlogging import Logger
from util.train import train

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

network = Baseline(logdir="test")

if use_cuda:
    network.cuda()

logger = Logger(network)

start = time.time()
train(net=network,
      num_epoch=150,
      train_set=train_set,
      batch_size=128,
      criterion=nn.CrossEntropyLoss(),
      logger=logger,
      test_set=test_set,
      learn_rate=0.001)

print(f"{round(time.time() - start, 2)}s")
print(accuracy(network, train_set, 128))
print(accuracy(network, test_set, 128))
