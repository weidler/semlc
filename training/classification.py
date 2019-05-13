import random

import numpy
import torch

import torchvision
from torch import nn
from torchvision.transforms import ToTensor, transforms

from model.network.benchmark import BaseClassificationCNN
from model.network.inhibition import InhibitionClassificationCNN
from training.train import train, accuracy

torch.random.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

baseline_network = BaseClassificationCNN()
inhibition_network = InhibitionClassificationCNN()
recurrent_inhibition_network = InhibitionClassificationCNN(inhibition_strategy="recurrent")

network = baseline_network

train(net=network,
      num_epoch=3,
      train_set=train_set,
      batch_size=16,
      criterion=nn.CrossEntropyLoss())

print(accuracy(network, train_set, 10))
