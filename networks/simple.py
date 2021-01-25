import math

import torch
import torchsummary
from layers.base import BaseSemLCLayer
from networks import BaseNetwork
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer


class Simple(BaseNetwork):

    def __init__(self, input_shape, n_classes: int, lateral_layer_function: BaseSemLCLayer = None, conv_one_init_std: float = 0.0001):
        super().__init__(input_shape=input_shape, lateral_layer_function=lateral_layer_function)

        # first convolution block
        self.conv_one = nn.Conv2d(self.input_channels, 64, kernel_size=(5, 5), stride=1, padding=2)  # output shape same
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape
        if self.lateral_layer_function is not None:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        self.pool_one = nn.MaxPool2d((3, 3), stride=(2, 2))  # output shape - 2 // 2
        self.bn_one = nn.BatchNorm2d(conv_one_out_size[-3])

        # second convolution block
        self.conv_two = nn.Conv2d(conv_one_out_size[-3], 64, kernel_size=(5, 5), stride=1, padding=2)  # output shape same
        self.bn_two = nn.BatchNorm2d(64)
        self.pool_conv_two = nn.MaxPool2d((3, 3), stride=(2, 2))  # output shape - 2 // 2
        self.drop_conv_two = nn.Dropout(0.2)

        # third convolution block
        self.conv_three = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1)  # output shape same

        # fourth convolution block
        self.conv_four = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1)  # output shape same
        self.pool_conv_four = nn.MaxPool2d((3, 3), stride=(2, 2))  # output shape - 2 // 2
        self.drop_conv_four = nn.Dropout(0.2)

        out_size = [math.ceil(((((((self.input_height
                                   - 2) / 2)   # first pooling
                                 - 2) / 2)  # second pooling
                               - 2) / 2)),  # fourth pooling
                    math.ceil(((((((self.input_width
                                   - 2) / 2)
                                 - 2) / 2)
                               - 2) / 2))]

        flattened_out_size = 32 * out_size[0] * out_size[1]
        self.classifier = nn.Sequential(
            nn.Linear(flattened_out_size, flattened_out_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(flattened_out_size // 2, flattened_out_size // 2 // 2),
            nn.ReLU(inplace=True),
            nn.Linear(flattened_out_size // 2 // 2, n_classes),
        )

        # weight initialization
        nn.init.normal_(self.conv_one.weight, 0, conv_one_init_std)
        nn.init.normal_(self.conv_two.weight, 0, 0.01)
        nn.init.normal_(self.conv_three.weight, 0, 0.04)
        nn.init.normal_(self.conv_four.weight, 0, 0.04)

        for module in self.classifier.children():
            if hasattr(module, "weight"):
                nn.init.normal_(module.weight, 0, 0.01)

    def forward(self, x: torch.Tensor):
        out_conv_one = self.conv_one(x)
        if self.lateral_layer_function is not None:
            out_conv_one = self.lateral_layer(out_conv_one)

        out_one = self.bn_one(self.pool_one(F.relu(out_conv_one)))
        out_two = self.drop_conv_two(self.pool_conv_two(self.bn_two(F.relu(self.conv_two(out_one)))))
        out_three = F.relu(self.conv_three(out_two))
        out_four = self.drop_conv_four(self.pool_conv_four(F.relu(self.conv_four(out_three))))

        flat = torch.flatten(out_four, 1)
        out = self.classifier(flat)

        return out

    def make_preferred_optimizer(self) -> Optimizer:
        return optim.Adam(self.parameters(), lr=0.001)


if __name__ == '__main__':
    sb_plus = Simple(input_shape=(3, 32, 32), n_classes=10)

    torchsummary.summary(sb_plus, sb_plus.input_shape, device="cpu")
