"""
A script for all main experiments that allows running multiple experiments of the same strategy
"""
import sys

from torch.utils.data import Subset

sys.path.append("./")

from networks import vgg19, vgg19_inhib

import torch

import torchvision
from torch import nn
from torchvision import transforms

from networks import BaselineCMap, Baseline, SingleShotInhibitionNetwork, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork
from utilities.train import train
from utilities.eval import accuracy

from utilities.log import Logger

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(strategy: str, iterations: int):
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

    for i in range(0, iterations):
        val_indices = list(range(int((i % 10) * len(trainval_set) / 10), int(((i % 10) + 1) * len(trainval_set) / 10)))
        train_indices = list(filter(lambda x: x not in val_indices, list(range(len(trainval_set)))))
        val_set = Subset(trainval_set, indices=val_indices)
        train_set = Subset(trainval_set, indices=train_indices)

        network = None
        if strategy == "baseline":
            network = Baseline()
        elif strategy == "cmap":
            network = BaselineCMap()
        elif strategy == "ss":
            network = SingleShotInhibitionNetwork(8, 0.2)
        elif strategy == "ss_self":
            network = SingleShotInhibitionNetwork(8, 0.2, self_connection=True)
        elif strategy == "ss_zeros":
            network = SingleShotInhibitionNetwork(8, 0.2, pad="zeros")
        elif strategy == "ss_full":
            network = SingleShotInhibitionNetwork([8, 8, 8, 8], [0.2, 0.2, 0.2, 0.2])
        elif strategy == "ss_freeze":
            network = SingleShotInhibitionNetwork(3, 0.1)
        elif strategy == "ss_freeze_self":
            network = SingleShotInhibitionNetwork(3, 0.1, self_connection=True)
        elif strategy == "ss_freeze_zeros":
            network = SingleShotInhibitionNetwork(3, 0.1, pad="zeros")
        elif strategy == "converged":
            network = ConvergedInhibitionNetwork(3, 0.1)
        elif strategy == "converged_self":
            network = ConvergedInhibitionNetwork(3, 0.1, self_connection=True)
        elif strategy == "converged_zeros":
            network = ConvergedInhibitionNetwork(3, 0.1, pad="zeros")
        elif strategy == "converged_full":
            network = ConvergedInhibitionNetwork([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1])
        elif strategy == "converged_cov_12":
            network = ConvergedInhibitionNetwork([3, 3], [0.1, 0.1])
        elif strategy == "converged_cov_123":
            network = ConvergedInhibitionNetwork([3, 3, 3], [0.1, 0.1, 0.1])
        elif strategy == "converged_freeze":
            network = ConvergedInhibitionNetwork(3, 0.2)  # toeplitz
        elif strategy == "converged_freeze_self":
            network = ConvergedInhibitionNetwork(3, 0.2, self_connection=True)
        elif strategy == "converged_freeze_zeros":
            network = ConvergedInhibitionNetwork(3, 0.2, pad="zeros")
        elif strategy == "converged_freeze_full":
            network = ConvergedInhibitionNetwork([3, 3, 3, 3], [0.2, 0.2, 0.2, 0.2])
        elif strategy == "parametric":
            network = ParametricInhibitionNetwork(3, 0.2)
        elif strategy == "parametric_self":
            network = ParametricInhibitionNetwork(3, 0.2, self_connection=True)
        elif strategy == "parametric_zeros":
            network = ParametricInhibitionNetwork(3, 0.2, pad="zeros")
        elif strategy == "parametric_full":
            network = ParametricInhibitionNetwork([3, 3, 3, 3], [0.2, 0.2, 0.2, 0.2])
        elif strategy == "parametric_cov_12":
            network = ParametricInhibitionNetwork([3, 3], [0.2, 0.2])
        elif strategy == "parametric_cov_123":
            network = ParametricInhibitionNetwork([3, 3, 3], [0.2, 0.2, 0.2])

        elif strategy == "vgg19":
            network = vgg19()
        elif strategy == "vgg19_inhib":
            network = vgg19_inhib()
        elif strategy == "vgg19_inhib_self":
            network = vgg19_inhib(self_connection=True)

        print(f"{network.__class__.__name__}_{i + 1}")
        print(network.features)

        if use_cuda:
            network.cuda()

        logger = Logger(network, experiment_code=f"{strategy}_{i}")

        train(net=network,
              num_epoch=180,
              train_set=train_set,
              batch_size=128,
              criterion=nn.CrossEntropyLoss(),
              logger=logger,
              val_set=val_set,
              # optimizer=SGD(networks.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),  # for VGG
              # learn_rate=0.05  # for VGG
              learn_rate=0.001,
              verbose=False)

        network.eval()
        logger.log(f"\nFinal Test Accuracy: {accuracy(network, test_set, 128)}")


if __name__ == '__main__':
    import argparse

    strategies = ["baseline", "cmap", "ss", "ss_freeze", "converged", "converged_self", "converged_freeze",
                  "converged_freeze_self", "parametric", "parametric_self", "vgg19", "vgg19_inhib", "vgg19_inhib_self",
                  "converged_freeze_zeros", "converged_zeros", "parametric_zeros", "ss_self", "ss_zeros",
                  "ss_freeze_self", "ss_freeze_zeros", "converged_full", "ss_full", "parametric_full",
                  "parametric_cov_12", "parametric_cov_123", "converged_cov_12", "converged_cov_123"]

    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", type=str, choices=strategies)
    parser.add_argument("-i", type=int, default=30, help="the number of iterations")
    args = parser.parse_args()

    run(args.strategy, args.i)
