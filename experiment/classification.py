import random

import numpy
import torch
import torchvision
from torch import nn
from torchvision import transforms

from experiment.eval import accuracy
from experiment.train import train
from model.network.alexnet_paper import ConvNet18, ConvNet11
from util.ourlogging import Logger

torch.random.manual_seed(12311)
numpy.random.seed(12311)
random.seed(12311)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(f"USE CUDA: {use_cuda}.")

transform = transforms.Compose([transforms.RandomCrop(24),
                                transforms.RandomVerticalFlip(),

                                # these transforms need to be in the end, s.t. the data loader produces (normed) tensors
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

train_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

# baseline_network = BaseClassificationCNN()
# inhibition_network = InhibitionClassificationCNN(learn_inhibition_weights=True)
# recurrent_inhibition_network = InhibitionClassificationCNN(inhibition_strategy="recurrent")
# alexnet = SmallAlexNet()
conv18 = ConvNet18()
conv11 = ConvNet11()

network = conv11

if use_cuda:
    network.cuda()

logger = Logger(network)

train(net=network,
      num_epoch=10,
      train_set=train_set,
      batch_size=128,
      criterion=nn.CrossEntropyLoss(),
      logger=logger,
      check_loss=100,
      # optimizer=custom_optimizer_conv18(network)
      learn_rate=0.001)

print(accuracy(network, test_set, 128))
