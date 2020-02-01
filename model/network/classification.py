from torch import nn as nn, nn

from model.inhibition_module import InhibitionModule
from model.alternative_inhibition_layers import Conv3DSingleShotInhibition, Conv3DRecurrentInhibition


class InhibitionClassificationCNN(nn.Module, InhibitionModule):

    def __init__(self, inhibition_strategy: str = "once", learn_inhibition_weights=False):
        super().__init__()
        self.inhibition_strategy = inhibition_strategy

        assert self.inhibition_strategy in ["once", "recurrent"]

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            Conv3DSingleShotInhibition(5, learn_weights=learn_inhibition_weights) if self.inhibition_strategy == "once"
                        else Conv3DRecurrentInhibition(5, learn_weights=learn_inhibition_weights),
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


class BaseClassificationCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
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