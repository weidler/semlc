import torch
import torchsummary
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer

from layers import BaseSemLCLayer
from networks import BaseNetwork


class Shallow(BaseNetwork):

    def __init__(self, input_shape, n_classes, lateral_layer_function: BaseSemLCLayer = None):
        super().__init__(input_shape, lateral_layer_function)

        # first convolution block -> output shape same
        self.conv_one = nn.Conv2d(self.input_channels, 64, kernel_size=5, stride=1, padding=2, )
        conv_one_out_size = self.conv_one(self.generate_random_input()).shape

        self.bn_one = nn.BatchNorm2d(conv_one_out_size[-3])
        self.pool_conv_one = nn.MaxPool2d((2, 2), stride=2)  # output shape 14

        if self.lateral_layer_function is not None:
            self.lateral_layer = self.lateral_layer_function(self.conv_one)
            self.lateral_layer.compile(conv_one_out_size[-2:])

        # second convolution block
        self.conv_two = nn.Conv2d(conv_one_out_size[-3],
                                  64, kernel_size=(3, 3), stride=1, padding=1)  # output shape same
        self.bn_two = nn.BatchNorm2d(64)
        self.pool_conv_two = nn.MaxPool2d((2, 2), stride=2)

        out_size = ((self.input_height // 2) // 2, (self.input_width // 2) // 2)

        self.classifier = nn.Sequential(
            nn.Linear(64 * out_size[0] * out_size[1], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
        )

        # WEIGHT INITIALIZATION
        nn.init.normal_(self.conv_one.weight, 0, 0.0001)
        nn.init.normal_(self.conv_two.weight, 0, 0.01)

    def forward(self, x: torch.Tensor):
        out_conv_one = self.conv_one(x)
        if self.lateral_layer_function is not None:
            out_conv_one = self.lateral_layer(out_conv_one)

        out_one = self.pool_conv_one(self.bn_one(F.relu(out_conv_one)))
        out_two = self.pool_conv_two(self.bn_two(F.relu(self.conv_two(out_one))))

        flat = torch.flatten(out_two, 1)
        out = self.classifier(flat)

        return out

    def make_preferred_optimizer(self) -> Optimizer:
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)

    def get_final_block1_layer(self):
        return self.pool_conv_one


if __name__ == '__main__':
    sb_plus = Shallow(input_shape=(3, 32, 32), n_classes=10)

    torchsummary.summary(sb_plus, sb_plus.input_shape, device="cpu")
