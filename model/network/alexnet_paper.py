from torch import nn
import torch
from torch.nn import Dropout

from model.inhibition_layer import SingleShotInhibition, ConvergedInhibition


class ConvNet18(nn.Module):

    def __init__(self, logdir=None, scope=[1,1,1], width=0, damp=0, inhibition_strategy: str = "once", learn_inhibition_weights=False, inhibition_depth=0):
        super().__init__()
        counter = 1
        self.logdir = logdir
        # self.inhibition_strategy = inhibition_strategy

        # assert self.inhibition_strategy in ["once", "recurrent"]

        self.features = nn.Sequential()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv1.weight, 0, 0.0001)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu1 = nn.ReLU(inplace=True)

        # rnorm1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv2.weight, 0, 0.01)

        self.relu2 = nn.ReLU(inplace=True)

        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # rnorm2 = nn.BatchNorm2d(3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        torch.nn.init.normal_(self.conv3.weight, 0, 0.01)

        self.relu3 = nn.ReLU(inplace=True)

        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)

        self.features.add_module("conv_1", self.conv1)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("relu_1", self.relu1)
        # self.features.add_module("rnorm_1", rnorm1)
        self.features.add_module("conv_2", self.conv2)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
                                     SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
                                     if self.inhibition_strategy == "once"
                                     else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("pool_2", self.pool2)
        # self.features.add_module("rnorm_2", rnorm2)
        self.features.add_module("conv_3", self.conv3)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_3", self.relu3)
        self.features.add_module("pool_3", self.pool3)

        # self.fc = nn.Linear(64 * 3 * 3, 10)
        self.fc = nn.Linear(64 * 2 * 2, 10)
        torch.nn.init.normal_(self.fc.weight, 0, 0.01)

        self.classifier = nn.Sequential(
            Dropout(),
            self.fc
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 64 * 3 * 3)
        x = x.view(x.size(0), 64 * 2 * 2)
        x = self.classifier(x)

        return x


class ConvNet11(nn.Module):

    def __init__(self, logdir=None, scope=[1,1,1,1], width=0, damp=0, inhibition_strategy: str = "once", learn_inhibition_weights=False, inhibition_depth=0):
        super().__init__()
        counter = 1
        self.logdir = logdir
        # self.inhibition_strategy = inhibition_strategy

        # assert self.inhibition_strategy in ["once", "recurrent"]

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
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("rnorm_1", rnorm1)
        self.features.add_module("conv_2", self.conv2)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("rnorm_2", rnorm2)
        self.features.add_module("pool_2", self.pool2)
        self.features.add_module("conv_3", self.conv3)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_3", self.relu3)
        self.features.add_module("conv_4", self.conv4)
        if counter <= inhibition_depth:
            self.features.add_module("inhib_{}".format(counter),
            SingleShotInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights)
            if self.inhibition_strategy == "once"
            else ConvergedInhibition(scope[counter-1], width, damp, learn_weights=learn_inhibition_weights))
            counter+=1
        self.features.add_module("relu_4", self.relu4)

        self.fc = nn.Linear(32 * 5 * 5, 10)
        torch.nn.init.normal_(self.fc.weight, 0, 0.01)

        self.classifier = nn.Sequential(
            # Dropout(),
            self.fc
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    net = ConvNet18()
    print(sum(p.numel() for p in net.parameters()))
