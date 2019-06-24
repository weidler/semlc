import random

import numpy
import torch

import torchvision
from torchvision import transforms

from util.eval import accuracy_with_confidence, accuracy
from model.network.alexnet_paper import InhibitionNetwork, Baseline, BaselineCMap, SingleShotInhibitionNetwork

# Random seeding is very important, since without the random cropping may be different
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

transform = transforms.Compose([transforms.RandomCrop(24),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

nets = [SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True) for i in range(1, 11)]

for i, net in enumerate(nets, 1):
    net.load_state_dict(torch.load(f"../saved_models/ss_freeze_{i}/SingleShotInhibitionNetwork_freeze_final.model"))

print(accuracy_with_confidence(nets, test_set, 128, 0.95))

