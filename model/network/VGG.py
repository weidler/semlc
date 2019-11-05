"""
Source: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py

"""

from typing import List

import torch
import torch.nn as nn

from model.inhibition_layer import ConvergedFrozenInhibition
from model.network.base import _BaseNetwork
from .utils import load_state_dict_from_url


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

from ..inhibition_module import InhibitionModule

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(_BaseNetwork, nn.Module):

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
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
    'EI': [64, 'I', 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(batch_norm=False, **kwargs):
    return VGG(make_layers(cfgs['A'], batch_norm=batch_norm), **kwargs)


def vgg13(batch_norm=False, **kwargs):
    return VGG(make_layers(cfgs['B'], batch_norm=batch_norm), **kwargs)


def vgg16(batch_norm=False, **kwargs):
    return VGG(make_layers(cfgs['D'], batch_norm=batch_norm), **kwargs)


def vgg16_inhib(batch_norm=False, num_classes=10, padding='circular', self_connection=False, **kwargs):
    inhib_layers = [ConvergedFrozenInhibition(scope=27,
                                              ricker_width=4, damp=0.12,
                                              in_channels=64, pad=padding, self_connection=self_connection)]
    return VGG(make_layers(cfgs['DI'], batch_norm=batch_norm, inhibition_layers=inhib_layers), num_classes=num_classes, **kwargs)


def vgg19(batch_norm=False, num_classes=10, **kwargs):
    return VGG(make_layers(cfgs['E'], batch_norm=batch_norm), num_classes=num_classes, **kwargs)


def vgg19_inhib(batch_norm=False, num_classes=10, padding='circular', self_connection=False, **kwargs):
    inhib_layers = [ConvergedFrozenInhibition(scope=27,
                                              ricker_width=4, damp=0.12,
                                              in_channels=64, pad=padding, self_connection=self_connection)]
    return VGG(make_layers(cfgs['EI'], batch_norm=batch_norm, inhibition_layers=inhib_layers), num_classes=num_classes, **kwargs)
