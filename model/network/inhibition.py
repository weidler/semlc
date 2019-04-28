import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import Inhibition
from model.inhibition_module import InhibitionModule


class InhibitionClassificationCNN(nn.Module, InhibitionModule):

    def __init__(self):
        super().__init__()

        self.layers = []
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.inhib1 = Inhibition(5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.inhib2 = Inhibition(5)

        self.layers.append(self.conv1)
        self.layers.append(self.inhib1)
        self.layers.append(self.conv2)
        self.layers.append(self.inhib2)

        self.classifying = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.inhib1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.inhib2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifying(x)

        return x

    def get_layers_for_visualization(self):
        return self.layers


if __name__ == "__main__":
    net = InhibitionClassificationCNN()
    net(torch.ones([1, 3, 32, 32]))
