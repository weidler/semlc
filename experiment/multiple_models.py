import sys
sys.path.append("../")

import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_paper import ConvNet11
from experiment.train import train
from experiment.eval import accuracy

from util.ourlogging import Logger


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

iterations = 10
for i in range(0, iterations):
    network = ConvNet11(logdir=f"baseline_{i+1}")

    if use_cuda:
        network.cuda()

    logger = Logger(network)

    train(net=network,
          num_epoch=160,
          train_set=train_set,
          batch_size=128,
          criterion=nn.CrossEntropyLoss(),
          logger=logger,
          test_set=test_set,
          learn_rate=0.001)

    network.eval()
    logger.log(f"acc: {accuracy(network, test_set, 128)}")
