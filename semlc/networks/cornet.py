"""Adapted from https://github.com/dicarlolab/CORnet"""

import abc
import math
from collections import OrderedDict

import torch
from torch import nn

from networks import BaseNetwork, BaseSemLCLayer
from layers.util import prepare_lc_builder


class Flatten(nn.Module):
    """Helper module for flattening input tensor to 1-D for the use in Linear modules."""

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """Helper module that stores the current tensor. Useful for accessing by name."""

    def forward(self, x):
        return x


class CORBlockS(nn.Module):
    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORBlockZ(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x


class BaseCORnet(BaseNetwork, abc.ABC):

    def forward(self, x):
        v1_latent = self.v1(x)
        v2_latent = self.v2(v1_latent)
        v4_latent = self.v4(v2_latent)
        it_latent = self.it(v4_latent)

        classes = self.classifier(it_latent)

        return classes

    def get_final_block1_layer(self):
        return self.v1.modules[-1]


class CORnetZ(BaseCORnet):

    def __init__(self, input_shape, n_classes, lateral_layer_function: BaseSemLCLayer = None):
        super().__init__(input_shape, lateral_layer_function)

        self.conv_one = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=7 // 2)
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape
        if self.lateral_layer_function is not None:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])
        else:
            self.lateral_layer = Identity()

        self.v1 = nn.Sequential(OrderedDict([
            ("conv1", self.conv_one),
            ('lateral', self.lateral_layer),
            ("nonlin", nn.ReLU(inplace=True)),
            ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ("output", Identity())  # for an easy access to this block's output
        ]))

        self.v2 = CORBlockZ(64, 128)
        self.v4 = CORBlockZ(128, 256)
        self.it = CORBlockZ(256, 512)

        self.classifier = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, n_classes)),
            ('output', Identity())
        ]))

        # weight initialization
        for m in [self.v1, self.v2, self.it, self.classifier]:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CORnetS(BaseCORnet):

    def __init__(self, input_shape, n_classes, lateral_layer_function: BaseSemLCLayer = None):
        super().__init__(input_shape, lateral_layer_function)

        self.conv_one = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape
        if self.lateral_layer_function is not None:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])
        else:
            self.lateral_layer = Identity()

        self.v1 = nn.Sequential(OrderedDict([  # this one is custom to save GPU memory
            ('conv1', self.conv_one),
            ('lateral', self.lateral_layer),
            ('norm1', nn.BatchNorm2d(64)),
            ('nonlin1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('nonlin2', nn.ReLU(inplace=True)),

            ('output', Identity())
        ]))

        self.v2 = CORBlockS(64, 128, times=2)
        self.v4 = CORBlockS(128, 256, times=4)
        self.it = CORBlockS(256, 512, times=2)

        self.classifier = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, n_classes)),
            ('output', Identity())
        ]))

        # weight initialization
        for m in [self.v1, self.v2, self.it, self.classifier]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # nn.Linear is missing here because I originally forgot
            # to add it during the training of this network
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    in_shape = (3, 64, 64)
    lateral_function = prepare_lc_builder("semlc", 3, 0.2)

    cors = CORnetS(input_shape=in_shape, n_classes=1000)
    cors(torch.rand((10, *in_shape)))

    corz = CORnetZ(input_shape=in_shape, n_classes=1000)
    corz((torch.rand((10, *in_shape))))
