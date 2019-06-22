from torch import nn
import torch
from torch.nn import Dropout

from model.inhibition_layer import SingleShotInhibition, ConvergedInhibition, ConvergedToeplitzFrozenInhibition
from model.inhibition_module import InhibitionModule


class Baseline(nn.Module):

    def __init__(self, logdir=None):
        super().__init__()
        self.logdir = logdir

        self.features = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.0001)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.rnorm1 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.01)
        self.relu2 = nn.ReLU(inplace=True)
        self.rnorm2 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.04)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(self.conv4.weight, 0, 0.04)
        self.relu4 = nn.ReLU(inplace=True)

        self.features.add_module("conv_1", self.conv1)
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("rnorm_1", self.rnorm1)
        self.features.add_module("conv_2", self.conv2)
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("rnorm_2", self.rnorm2)
        self.features.add_module("pool_2", self.pool2)
        self.features.add_module("conv_3", self.conv3)
        self.features.add_module("relu_3", self.relu3)
        self.features.add_module("conv_4", self.conv4)
        self.features.add_module("relu_4", self.relu4)

        self.classifier = nn.Linear(32 * 5 * 5, 10)
        torch.nn.init.normal_(self.classifier.weight, 0, 0.01)

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

        rnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.01)

        self.relu2 = nn.ReLU(inplace=True)

        rnorm2 = nn.BatchNorm2d(64)

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
                                     SingleShotInhibition(scope=scope[counter-1], ricker_width=width, damp=damp,
                                                          learn_weights=False)
                                     if self.inhibition_strategy == "once"
                                     else SingleShotInhibition(scope=scope[counter-1], ricker_width=width, damp=damp,
                                                          learn_weights=True)
                                     if self.inhibition_strategy == "once_learned"
                                     else ConvergedInhibition(scope=scope[counter-1], ricker_width=width, damp=damp,
                                                              in_channels=64)
                                     if self.inhibition_strategy == "converged"
                                     else ConvergedToeplitzFrozenInhibition(scope=scope[counter-1],
                                                                            ricker_width=width, damp=damp,
                                                                            in_channels=64))
            counter += 1
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("rnorm_1", rnorm1)
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
        self.features.add_module("rnorm_2", rnorm2)
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
    net = InhibitionNetwork(scope=[13, 13, 13, 13], width=3, damp=0.12, inhibition_depth=1)
    net2 = InhibitionNetwork(scope=[13, 13, 13, 13], width=3, damp=0.12, inhibition_depth=2, inhibition_strategy="converged")
    net3 = InhibitionNetwork(scope=[13, 13, 13, 13], width=3, damp=0.12, inhibition_depth=3, inhibition_strategy="toeplitz")
    # print(sum(p.numel() for p in net.parameters()))
    print(net.features)
    print(net2.features)
    print(net3.features)
