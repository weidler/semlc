import random


import numpy
import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.classification import InhibitionClassificationCNN, BaseClassificationCNN
from experiment.train import train
from experiment.eval import accuracy

from util.logging import Logger

torch.random.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

baseline_network = BaseClassificationCNN()
inhibition_network = InhibitionClassificationCNN(learn_inhibition_weights=True)
recurrent_inhibition_network = InhibitionClassificationCNN(inhibition_strategy="recurrent")

network = recurrent_inhibition_network
logger = Logger(network)

train(net=network,
      num_epoch=20,
      train_set=train_set,
      batch_size=16,
      criterion=nn.CrossEntropyLoss(),
      logger=logger)

print(accuracy(network, test_set, 16))
