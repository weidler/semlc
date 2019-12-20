import sys

import torchsummary
from torch.utils.data import Subset

sys.path.append("../")

from torch.optim import SGD

from model.network.VGG import vgg19, vgg19_inhib

import torch

import torchvision
from torch import nn
from torchvision import transforms

from model.network.alexnet_cifar import BaselineCMap, Baseline, SingleShotInhibitionNetwork, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork
from util.train import train
from util.eval import accuracy

from util.ourlogging import Logger

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
    trainval_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10("../data/cifar10/", train=False, download=True, transform=transform)

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
            network = SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False)
        elif strategy == "ss_self":
            network = SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False, self_connection=True)
        elif strategy == "ss_zeros":
            network = SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False, pad="zeros")
        elif strategy == "ss_full":
            network = SingleShotInhibitionNetwork([63, 63, 63, 31], [8, 8, 8, 8], [0.2, 0.2, 0.2, 0.2], freeze=False,
                                                  coverage=4)
        elif strategy == "ss_freeze":
            network = SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True)
        elif strategy == "ss_freeze_self":
            network = SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True, self_connection=True)
        elif strategy == "ss_freeze_zeros":
            network = SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True, pad="zeros")
        elif strategy == "converged":
            network = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False)
        elif strategy == "converged_self":
            network = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, self_connection=True)
        elif strategy == "converged_zeros":
            network = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, pad="zeros")
        elif strategy == "converged_full":
            network = ConvergedInhibitionNetwork([27, 27, 27, 27], [3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1], freeze=False,
                                                 coverage=4)
        elif strategy == "converged_freeze":
            network = ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True)  # toeplitz
        elif strategy == "converged_freeze_self":
            network = ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True, self_connection=True)
        elif strategy == "converged_freeze_zeros":
            network = ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True, pad="zeros")
        elif strategy == "converged_freeze_full":
            # TODO not working yet due to hard coded in_channels in layer instantiation!
            network = ConvergedInhibitionNetwork([45, 45, 45, 23], [3, 3, 3, 3], [0.2, 0.2, 0.2, 0.2], freeze=True,
                                                 coverage=4)
        elif strategy == "parametric":
            network = ParametricInhibitionNetwork([45], 3, 0.2)
        elif strategy == "parametric_self":
            network = ParametricInhibitionNetwork([45], 3, 0.2, self_connection=True)
        elif strategy == "parametric_zeros":
            network = ParametricInhibitionNetwork([45], 3, 0.2, pad="zeros")
        elif strategy == "parametric_full":
            network = ParametricInhibitionNetwork([63, 63, 63, 31], [3, 3, 3, 3], [0.2, 0.2, 0.2, 0.2], coverage=4)
        elif strategy == "parametric_12":
            network = ParametricInhibitionNetwork([63, 63], [3, 3], [0.2, 0.2], coverage=2)
        elif strategy == "parametric_123":
            network = ParametricInhibitionNetwork([63, 63, 63], [3, 3, 3], [0.2, 0.2, 0.2], coverage=3)

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
              # optimizer=SGD(network.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4),
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
                  "parametric_12", "parametric_123"]

    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", type=str, choices=strategies)
    parser.add_argument("-i", type=int, default=30)
    args = parser.parse_args()

    run(args.strategy, args.i)
