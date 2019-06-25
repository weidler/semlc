import random

import numpy
import torch
import torchvision
from torchvision import transforms

from model.network.alexnet_paper import SingleShotInhibitionNetwork, BaselineCMap, Baseline, ConvergedInhibitionNetwork
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
nets = [ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1) for i in range(1, 11)]
# nets = [SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True) for i in range(1, 11)]
# nets = [SingleShotInhibitionNetwork([63], 3, 0.1, freeze=False) for i in range(1, 11)]
# nets = [Baseline() for i in range(1, 11)]
logdir = "converged"
# added_loc = "inhib_3-3/"
added_loc = ""
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
    filename =  f"../final_results/{logdir}/{logdir}_{i}/{added_loc}" \
                f"{net.__class__.__name__ + (f'_freeze' if hasattr(net, 'freeze') and net.freeze else '')}_best.model"
    print(f"Loading {filename}")
    net.load_state_dict(torch.load(filename))

print(accuracy_with_confidence(nets, test_set, 128, 0.95))
