import random

import numpy
import torch
import torchvision
from torchvision import transforms

from model.network.alexnet_paper import SingleShotInhibitionNetwork, BaselineCMap, Baseline
from util.eval import accuracy_with_confidence

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


# SET UP NETS AND SETTINGS
nets = [SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True) for i in range(1, 11)]
# nets = [Baseline() for i in range(1, 11)]
logdir = "ss_freeze"
random_transform_test = True


# LOAD TEST DATA
if random_transform_test:
    transform = transforms.Compose([transforms.RandomCrop(24),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
else:
    transform = transforms.Compose([transforms.CenterCrop(24),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)


# EVALUATE
for i, net in enumerate(nets, 1):
    net.load_state_dict(torch.load(
        f"../saved_models/{logdir}_{i}/"
        f"{net.__class__.__name__ + (f'_freeze' if hasattr(net, 'freeze') and net.freeze else '')}_final.model"))

print(accuracy_with_confidence(nets, test_set, 128, 0.95))
