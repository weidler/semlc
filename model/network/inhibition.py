import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import Inhibition, RecurrentInhibition
from model.inhibition_module import InhibitionModule


class InhibitionClassificationCNN(nn.Module, InhibitionModule):

    def __init__(self, inhibition_strategy: str = "once", learn_inhibition_weights=False):
        super().__init__()

        assert inhibition_strategy in ["once", "recurrent"]

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            Inhibition(5, learn_weights=learn_inhibition_weights) if inhibition_strategy == "once"
                        else RecurrentInhibition(5, learn_weights=learn_inhibition_weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)

        return x

    def get_layers_for_visualization(self):
        return self.layers


if __name__ == "__main__":
    net = InhibitionClassificationCNN()
    net(torch.ones([1, 3, 32, 32]))
