"""Tests the accuracy on a given layers or an average over a set of models"""

import random
import sys

from torch.utils.data import DataLoader

from networks import vgg19, vgg19_inhib

sys.path.append("./")
import numpy
import torch
import torchvision
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

from networks import SingleShotInhibitionNetwork, BaselineCMap, Baseline, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork
from utilities.eval import accuracies_from_list, accuracy_from_data_loader

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

keychain = "./output/keychain.txt"
model_path = "./output/"

df = pd.read_csv(keychain, sep="\t", names=['id', 'group', 'layers', 'datetime'])

# SET UP NETS AND SETTINGS

num_nets = 10


# extend this for future experiments
all_nets = {
    # baselines
    'baseline': [Baseline() for i in range(1, num_nets + 1)],
    'cmap': [BaselineCMap() for i in range(1, num_nets + 1)],

    # ssi
    'ss': [SingleShotInhibitionNetwork(8, 0.2) for i in range(1, num_nets + 1)],
    'ss_freeze': [SingleShotInhibitionNetwork(3, 0.1) for i in range(1, num_nets + 1)],
    'ss_freeze_zeros': [SingleShotInhibitionNetwork(3, 0.1, pad="zeros") for i in range(1, num_nets + 1)],
    'ss_freeze_self': [SingleShotInhibitionNetwork(3, 0.1, self_connection=True) for i in range(1, num_nets + 1)],
    'ss_zeros': [SingleShotInhibitionNetwork(8, 0.2, pad="zeros") for i in range(1, num_nets + 1)],
    'ss_self': [SingleShotInhibitionNetwork(3, 0.1, self_connection=True) for i in range(1, num_nets + 1)],

    # converged
    'converged': [ConvergedInhibitionNetwork(3, 0.1) for i in range(1, num_nets + 1)],
    'converged_freeze': [ConvergedInhibitionNetwork(3, 0.2) for i in range(1, num_nets + 1)],
    'converged_zeros': [ConvergedInhibitionNetwork(3, 0.1, pad="zeros") for i in range(1, num_nets + 1)],
    'converged_freeze_zeros': [ConvergedInhibitionNetwork(3, 0.2, pad="zeros") for i in range(1, num_nets + 1)],
    'converged_self': [ConvergedInhibitionNetwork(3, 0.1, self_connection=True) for i in range(1, num_nets + 1)],
    'converged_freeze_self': [ConvergedInhibitionNetwork(3, 0.2, self_connection=True) for i in range(1, num_nets + 1)],
    'converged_cov_12': [ConvergedInhibitionNetwork([3, 3], [0.1, 0.1]) for i in range(1, num_nets + 1)],
    'converged_cov_123': [ConvergedInhibitionNetwork([3, 3, 3], [0.1, 0.1, 0.1]) for i in range(1, num_nets + 1)],
    'converged_full': [ConvergedInhibitionNetwork([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1]) for i in range(1, num_nets + 1)],
    'converged_full_best': [ConvergedInhibitionNetwork([3, 10, 3, 10], [0.12, 0.1, 0.14, 0.12]) for i in range(1, num_nets + 1)],


    # parametric
    'parametric': [ParametricInhibitionNetwork(3, 0.2) for i in range(1, num_nets + 1)],
    'parametric_zeros': [ParametricInhibitionNetwork(3, 0.2, pad="zeros") for i in range(1, num_nets + 1)],
    'parametric_self': [ParametricInhibitionNetwork(3, 0.2, self_connection=True) for i in range(1, num_nets + 1)],
    'parametric_12': [ParametricInhibitionNetwork([3, 3], [0.2, 0.2]) for i in range(1, num_nets + 1)],
    'parametric_123': [ParametricInhibitionNetwork([3, 3, 3], [0.2, 0.2, 0.2]) for i in range(1, num_nets + 1)],

    # vgg
    'vgg19': vgg19(),
    'vgg19_inhib': vgg19_inhib()
}

strategies = all_nets.keys()
analyse_strats = ['ss_freeze_self']  # ['cmap', 'ss', 'ss_freeze', 'converged_freeze', 'converged']

# test on both the augmented and non-augmented test set
for random_transform_test in [False, True]:
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

    test_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transform)

    # EVALUATE
    for strategy in strategies:
        if strategy in all_nets.keys():
            if strategy not in analyse_strats:
                print(f"Skipping strategy {strategy}.")
                continue

            accuracies = []
            # match the exact strategy followed by _ and 1 or optional 2 digits
            filenames = df[df['group'].str.match(rf'{strategy}_\d\d?')]['id']
            print(strategy, len(filenames))

            for i, row in tqdm(enumerate(filenames), disable=True):
                filename = f"{model_path}{row}_best.layers"
                all_nets[strategy][i].load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
                data_loader = DataLoader(test_set, batch_size=128,
                                         shuffle=False, num_workers=0)
                acc = accuracy_from_data_loader(all_nets[strategy][i], data_loader)
                print(acc)
                accuracies.append(acc)
            print(f"\nLoaded {len(filenames)} files for strategy {strategy}.")
            if len(filenames):
                print(accuracies)
                print(f"{strategy}{'[wA]' if random_transform_test else ''}: {accuracies_from_list(accuracies)}")
