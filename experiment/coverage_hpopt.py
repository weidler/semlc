"""Hyper parameter optimization script exploring the coverage of LC layers"""

import sys
from torch.utils.data import Subset

sys.path.append("./")

import torch
import numpy as np

import random
import pandas as pd

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_cifar import ConvergedInhibitionNetwork
from util.train import train
from util.eval import accuracy, accuracy_with_confidence

from util.ourlogging import Logger

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

range_scope = np.array([[9, 27, 45, 63],
                        [9, 27, 45, 63],
                        [9, 27, 45, 63],
                        [7, 17, 25, 31],
                        ])
range_ricker_width = [3, 4, 6, 8, 10]
range_damp = [0.1, 0.12, 0.14, 0.16, 0.2]


def run(strategy: str, runs: int, iterations: int):
    crop = 32 if "vgg" in strategy else 24
    padding = 4 if "vgg" in strategy else None

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print(f"USE CUDA: {use_cuda}.")

    # transformation
    transform = transforms.Compose([transforms.RandomCrop(crop, padding),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    # load data
    trainval_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=transform)

    df = pd.DataFrame(columns=['config', 'val_acc', 'val_acc_wA'])

    for r in range(0, runs):
        config = f"{strategy}_{r}"
        coverage = 4
        scopes = [random.choice(range_scope[i]) for i in range(coverage)]
        widths = [random.choice(range_ricker_width) for _ in range(coverage)]
        damps = [random.choice(range_damp) for _ in range(coverage)]
        print('scopes:', scopes)
        print('widths:', widths)
        print('damps:', damps)

        networks = []
        for i in range(0, iterations):
            val_indices = list(
                range(int((i % 10) * len(trainval_set) / 10), int(((i % 10) + 1) * len(trainval_set) / 10)))
            train_indices = list(filter(lambda x: x not in val_indices, list(range(len(trainval_set)))))
            val_set = Subset(trainval_set, indices=val_indices)
            train_set = Subset(trainval_set, indices=train_indices)

            if strategy == "converged_full_hp":
                network = ConvergedInhibitionNetwork(widths, damps)
                networks.append(network)

            print(f"{networks[i].__class__.__name__}_{i + 1}")
            print(networks[i].features)

            if use_cuda:
                networks[i].cuda()

            logger = Logger(networks[i], experiment_code=f"{strategy}_{r}_{i}")

            train(net=networks[i],
                  num_epoch=1,
                  train_set=train_set,
                  batch_size=128,
                  criterion=nn.CrossEntropyLoss(),
                  logger=logger,
                  val_set=val_set,
                  learn_rate=0.001,
                  verbose=False)

            networks[i].eval()
            logger.log(f"\nFinal Test Accuracy: {accuracy(networks[i], test_set, 128)}")

        entry = {'config': config}
        for random_transform_test in [True, False]:
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

            acc = accuracy_with_confidence(networks, test_set, 128, 0.95)
            print(f"{strategy}{'[wA]' if random_transform_test else ''}: {acc}")
            if random_transform_test:
                entry['val_acc_wA'] = acc
            else:
                entry['val_acc'] = acc

        print('config:', entry)
        df = df.append(entry, ignore_index=True)
        df.to_csv('../output/hp_opt.csv', index=False)


if __name__ == '__main__':
    import argparse

    strategies = ["converged_full_hp"]

    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", type=str, choices=strategies)
    parser.add_argument("-i", type=int, default=10, help="no of iterations per run")
    parser.add_argument("-r", type=int, default=30, help="no of runs")
    args = parser.parse_args()

    run(args.strategy, args.r, args.i)
