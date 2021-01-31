import math

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer

from layers.base import BaseSemLCLayer
from networks import BaseNetwork


class AlexNet(BaseNetwork):

    def __init__(self, input_shape, n_classes: int, lateral_layer_function: BaseSemLCLayer = None):
        super().__init__(input_shape=input_shape, lateral_layer_function=lateral_layer_function)

        self.layer_output_channels = [64, 64, 64, 32]

        self.conv_one = nn.Conv2d(self.input_channels, 64, kernel_size=(5, 5), stride=1, padding=2)  # output shape same
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape
        if self.lateral_layer_function is not None:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        self.pool_one = nn.MaxPool2d((3, 3), stride=(2, 2))
        self.bn_one = nn.BatchNorm2d(conv_one_out_size[-3])

        self.conv_two = nn.Conv2d(conv_one_out_size[-3], self.layer_output_channels[1], kernel_size=5, stride=1, padding=2)
        self.bn_two = nn.BatchNorm2d(self.layer_output_channels[1])
        self.pool_two = nn.AvgPool2d((3, 3), stride=(2, 2))

        self.conv_three = nn.Conv2d(self.layer_output_channels[1], self.layer_output_channels[2], kernel_size=3, stride=1, padding=1)
        self.conv_four = nn.Conv2d(self.layer_output_channels[2], self.layer_output_channels[3], kernel_size=3, stride=1, padding=1)

        # WEIGHT INITIALIZATION
        nn.init.normal_(self.conv_one.weight, 0, 0.0001)
        nn.init.normal_(self.conv_two.weight, 0, 0.01)
        nn.init.normal_(self.conv_three.weight, 0, 0.04)
        nn.init.normal_(self.conv_four.weight, 0, 0.04)

        self.conv_out = [math.ceil((((self.input_height - 2) / 2) - 2) / 2),
                         math.ceil((((self.input_width - 2) / 2) - 2) / 2)]

        # FULL CONNECTED
        self.classifier = nn.Linear(self.layer_output_channels[3] * self.conv_out[0] * self.conv_out[1], n_classes)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x):
        x = self.extract_features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def make_preferred_optimizer(self) -> Optimizer:
        return optim.Adam(self.parameters(), lr=0.001)

    def extract_features(self, x):
        out_conv_one = self.conv_one(x)
        if self.lateral_layer_function is not None:
            out_conv_one = self.lateral_layer(out_conv_one)

        block_one = self.bn_one(self.pool_one(F.relu(out_conv_one)))
        block_two = self.pool_two(self.bn_two(F.relu(self.conv_two(block_one))))
        block_three = F.relu(self.conv_three(block_two))
        block_four = F.relu(self.conv_four(block_three))

        return block_four

    def get_final_block1_layer(self):
        return self.bn_one