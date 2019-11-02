import random
import time

import numpy
import torch
import torchvision
from torch import nn
from torchvision import transforms

from model.network.VGG import vgg16, vgg19_inhib
from model.network.alexnet_paper import Baseline, ConvergedInhibitionNetwork, SingleShotInhibitionNetwork, BaselineCMap
from util.eval import accuracy
from util.ourlogging import Logger
from util.train import train

torch.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

transform = transforms.Compose([transforms.RandomCrop(32, 4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

# network = Baseline(logdir="test")
# network = ConvergedInhibitionNetwork(scopes=[27], width=3, damp=0.1, freeze=True, inhibition_start=1, inhibition_end=1, logdir="test")
# network = ConvNet13(logdir="ConvNet13")
# network = BaselineCMap()
# network = vgg16_inhib()
network = vgg19_inhib()

if use_cuda:
    network.cuda()

logger = Logger(network)

logger.describe_network()

start = time.time()
train(net=network,
      num_epoch=300,
      train_set=train_set,
      batch_size=128,
      criterion=nn.CrossEntropyLoss(),
      logger=logger,
      val_set=test_set,
      learn_rate=0.01)

print(f"{round(time.time() - start, 2)}s")
print(accuracy(network, train_set, 128))
print(accuracy(network, test_set, 128))
