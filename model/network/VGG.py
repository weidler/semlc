"""Modified from Cheng-Yang Fu's implementation (https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py)
 and PyTorch's (https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)"""

from typing import List

import math
import torch
import torch.nn as nn

from model.inhibition_layer import ConvergedFrozenInhibition
from model.network.base import BaseNetwork


__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg16_inhib', 'vgg19', 'vgg19_inhib',
]

from ..inhibition_module import InhibitionModule


class VGG(BaseNetwork, nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        inhibition_layers = [layer for layer in self.features.children() if isinstance(layer, InhibitionModule)]
        self.is_circular = [layer.is_circular for layer in inhibition_layers]
        self.self_connection = [layer.self_connection for layer in inhibition_layers]
        self.damp = [layer.damp for layer in inhibition_layers]
        self.width = [layer.width for layer in inhibition_layers]
        self.scopes = [layer.scope for layer in inhibition_layers]

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # comment back in and replace with x.view for ImageNet
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


def make_layers(cfg, inhibition_layers: List[nn.Module]=None, batch_norm=False):
    layers = []
    in_channels = 3
    inhib_counter = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'I':
            layers += [inhibition_layers[inhib_counter]]
            inhib_counter += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'DI': [64, 'I', 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'EI': [64, 'I', 64, 'I', 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(batch_norm=False):
    return VGG()


def vgg13(batch_norm=False):
    return VGG()


def vgg16(batch_norm=False):
    return VGG()


def vgg16_inhib(batch_norm=False, num_classes=10, padding='circular', self_connection=False):
    inhib_layers = [ConvergedFrozenInhibition(in_channels=64, ricker_width=4, damp=0.12, pad=padding,
                                              self_connection=self_connection)]
    return VGG()


def vgg19(batch_norm=False, num_classes=10):
    return VGG()


def vgg19_inhib(batch_norm=False, num_classes=10, padding='circular', self_connection=False):
    inhib_layers = [ConvergedFrozenInhibition(in_channels=64, ricker_width=4, damp=0.12, pad=padding,
                                              self_connection=self_connection),
                    ConvergedFrozenInhibition(in_channels=64, ricker_width=4, damp=0.12, pad=padding,
                                              self_connection=self_connection)
                    ]
    return VGG()
