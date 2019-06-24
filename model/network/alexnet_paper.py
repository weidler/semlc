from typing import List, Dict

import torch
from torch import nn

from model.inhibition_layer import SingleShotInhibition, ConvergedInhibition, ConvergedToeplitzFrozenInhibition


class _AlexNetBase(nn.Module):

    def __init__(self, logdir=None):
        super().__init__()
        self.logdir = logdir

        self.features = nn.Sequential()

        # CONVOLUTION 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv1.weight, 0, 0.0001)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnorm1 = nn.BatchNorm2d(64)

        # CONVOLUTION 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        self.relu2 = nn.ReLU(inplace=True)
        self.bnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # CONVOLUTION 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv3.weight, 0, 0.04)
        self.relu3 = nn.ReLU(inplace=True)

        # CONVOLUTION 4
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv4.weight, 0, 0.04)
        self.relu4 = nn.ReLU(inplace=True)

        # FULL CONNECTED
        self.classifier = nn.Linear(32 * 5 * 5, 10)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x

    def build_module(self, inhibition_layers: Dict[str, nn.Module]):
        self.features.add_module("conv_1", self.conv1)
        if "inhib_1" in inhibition_layers.keys():
            self.features.add_module("inhib_1", inhibition_layers["inhib_1"])
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("bnorm_1", self.bnorm1)

        self.features.add_module("conv_2", self.conv2)
        if "inhib_2" in inhibition_layers.keys():
            self.features.add_module("inhib_2", inhibition_layers["inhib_2"])
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("bnorm_2", self.bnorm2)
        self.features.add_module("pool_2", self.pool2)

        self.features.add_module("conv_3", self.conv3)
        if "inhib_3" in inhibition_layers.keys():
            self.features.add_module("inhib_3", inhibition_layers["inhib_3"])
        self.features.add_module("relu_3", self.relu3)

        self.features.add_module("conv_4", self.conv4)
        if "inhib_4" in inhibition_layers.keys():
            self.features.add_module("inhib_4", inhibition_layers["inhib_4"])
        self.features.add_module("relu_4", self.relu4)


class Baseline(_AlexNetBase):

    def __init__(self, logdir=None):
        super().__init__(logdir=logdir)
        self.build_module({})


class BaselineCMap(_AlexNetBase):

    def __init__(self, logdir=None):
        super().__init__(logdir=logdir)

        self.cnorm1 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)
        self.cnorm2 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)

        self.features.add_module("conv_1", self.conv1)
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("cnorm_1", self.cnorm1)
        self.features.add_module("bnorm_1", self.bnorm1)

        self.features.add_module("conv_2", self.conv2)
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("cnorm_2", self.cnorm2)
        self.features.add_module("bnorm_2", self.bnorm2)
        self.features.add_module("pool_2", self.pool2)

        self.features.add_module("conv_3", self.conv3)
        self.features.add_module("relu_3", self.relu3)

        self.features.add_module("conv_4", self.conv4)
        self.features.add_module("relu_4", self.relu4)


class SingleShotInhibitionNetwork(_AlexNetBase):

    def __init__(self, scopes: List[int], width: float, damp: float, freeze=True, inhibition_start=1, inhibition_end=1,
                 logdir=None):
        super().__init__(logdir=logdir)

        if len(scopes) != inhibition_end - inhibition_start + 1:
            raise ValueError(f"Inconsistent number of given scopes ({len(scopes)}) and desired inhibition start/end "
                             f"({inhibition_start}/{inhibition_end}).")

        self.scopes = scopes
        self.width = width
        self.damp = damp

        self.freeze = freeze
        self.inhibition_start = inhibition_start
        self.inhibition_end = inhibition_end

        inhibition_layers = {}
        for i in range(inhibition_start, inhibition_end + 1):
            inhibition_layers.update(
                {f"inhib_{i}": SingleShotInhibition(scope=scopes[i - 1], ricker_width=width, damp=damp,
                                                    learn_weights=not freeze)})

        self.build_module(inhibition_layers)


class ConvergedInhibitionNetwork(_AlexNetBase):

    def __init__(self, scopes: List[int], width: float, damp: float, freeze=True, inhibition_start=1, inhibition_end=1,
                 logdir=None):
        super().__init__(logdir=logdir)

        #if len(scopes) != inhibition_end - inhibition_start + 1:
        #    raise ValueError(f"Inconsistent number of given scopes ({len(scopes)}) and desired inhibition start/end "
        #                     f"({inhibition_start}/{inhibition_end}).")

        self.scopes = scopes
        self.width = width
        self.damp = damp

        self.freeze = freeze
        self.inhibition_start = inhibition_start
        self.inhibition_end = inhibition_end

        inhibition_layers = {}
        for i in range(inhibition_start, inhibition_end + 1):
            inhibition_layers.update(
                {f"inhib_{i}": ConvergedInhibition(scope=scopes[i - 1], ricker_width=width, damp=damp,
                                                   in_channels=64) if not self.freeze else
                ConvergedToeplitzFrozenInhibition(scope=scopes[i - 1],
                                                  ricker_width=width, damp=damp,
                                                  in_channels=64)})

        self.build_module(inhibition_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


class InhibitionNetwork(nn.Module):

    def __init__(self, scope, width, damp, logdir=None, inhibition_strategy: str = "once",
                 inhibit_start=1, inhibition_depth=0):
        super().__init__()

        counter = inhibit_start
        self.logdir = logdir
        self.inhibition_strategy = inhibition_strategy

        assert self.inhibition_strategy in ["once", "once_learned", "converged", "toeplitz"]

        self.features = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.0001)

        self.relu1 = nn.ReLU(inplace=True)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        bnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.01)

        self.relu2 = nn.ReLU(inplace=True)

        bnorm2 = nn.BatchNorm2d(64)

        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # this should be a locally connected layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.04)

        self.relu3 = nn.ReLU(inplace=True)

        # this should be a locally connected layer
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.04)

        self.relu4 = nn.ReLU(inplace=True)

        self.features.add_module("conv_1", self.conv1)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     SingleShotInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                          learn_weights=False)
                                     if self.inhibition_strategy == "once"
                                     else SingleShotInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                               learn_weights=True)
                                     if self.inhibition_strategy == "once_learned"
                                     else ConvergedInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                              in_channels=64)
                                     if self.inhibition_strategy == "converged"
                                     else ConvergedToeplitzFrozenInhibition(scope=scope[counter - 1],
                                                                            ricker_width=width, damp=damp,
                                                                            in_channels=64))
            counter += 1
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("bnorm_1", bnorm1)
        self.features.add_module("conv_2", self.conv2)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     SingleShotInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                          learn_weights=self.learn_weights)
                                     if self.inhibition_strategy == "once"
                                     else ConvergedInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                              learn_weights=self.learn_weights, in_channels=64)
                                     if self.inhibition_strategy == "converged"
                                     else ConvergedToeplitzFrozenInhibition(scope=scope[counter - 1],
                                                                            ricker_width=width, damp=damp,
                                                                            in_channels=64))
            counter += 1
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("bnorm_2", bnorm2)
        self.features.add_module("pool_2", self.pool2)
        self.features.add_module("conv_3", self.conv3)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     SingleShotInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                          learn_weights=self.learn_weights)
                                     if self.inhibition_strategy == "once"
                                     else ConvergedInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                              learn_weights=self.learn_weights, in_channels=64)
                                     if self.inhibition_strategy == "converged"
                                     else ConvergedToeplitzFrozenInhibition(scope=scope[counter - 1],
                                                                            ricker_width=width, damp=damp,
                                                                            in_channels=64))
            counter += 1
        self.features.add_module("relu_3", self.relu3)
        self.features.add_module("conv_4", self.conv4)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     SingleShotInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                          learn_weights=self.learn_weights)
                                     if self.inhibition_strategy == "once"
                                     else ConvergedInhibition(scope=scope[counter - 1], ricker_width=width, damp=damp,
                                                              learn_weights=self.learn_weights, in_channels=32)
                                     if self.inhibition_strategy == "converged"
                                     else ConvergedToeplitzFrozenInhibition(scope=scope[counter - 1],
                                                                            ricker_width=width, damp=damp,
                                                                            in_channels=32))
            counter += 1
        self.features.add_module("relu_4", self.relu4)

        self.fc = nn.Linear(32 * 5 * 5, 10)
        torch.nn.init.normal_(self.fc.weight, 0, 0.01)

        self.classifier = nn.Sequential(
            self.fc
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    net = ConvergedInhibitionNetwork(scopes=[27, 27, 27], width=3, damp=0.1, freeze=True, inhibition_start=2,
                                     inhibition_end=2, logdir="test")
    print(net.features)
