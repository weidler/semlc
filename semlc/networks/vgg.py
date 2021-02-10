"""Credit to https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py"""

import torch.nn as nn
from networks import BaseNetwork
from layers import BaseSemLCLayer
from torch import optim

cfg = {
    'VGG11': [64, 'L', 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'L', 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'L', 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'L', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(BaseNetwork):
    def __init__(self, input_shape, vgg_name, lateral_layer_function: BaseSemLCLayer = None):
        super().__init__(input_shape, lateral_layer_function)
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            if x == 'L':
                if self.lateral_layer is not None:
                    layers += self.lateral_layer
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def make_preferred_optimizer(self):
        return optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)


class VGG11(VGG):

    def __init__(self, input_shape):
        super().__init__(input_shape, "VGG11")


class VGG16(VGG):

    def __init__(self, input_shape):
        super().__init__(input_shape, "VGG16")
