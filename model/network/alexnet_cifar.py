from typing import List, Dict, Union

import torch
from torchsummary import summary
from torch import nn

from model.semantic_layers import SingleShotSemLC, ConvergedFrozenSemLC, ConvergedSemLC, \
    ParametricSemLC
from model.network.base import BaseNetwork, BaseNetwork


class _AlexNetBase(BaseNetwork):
    """The abstract Baseline class for CIFAR-10 reconstructed from https://code.google.com/archive/p/cuda-convnet/"""

    def __init__(self, widths: List[int] = None, damps: List[float] = None, strategy: str = None, optim: str = None,
                 self_connection: bool = False, pad: str = "circular"):
        super().__init__(widths, damps, strategy, optim, self_connection, pad)

        self.features = nn.Sequential()
        self.layer_output_channels = [64, 64, 64, 32]

        # CONVOLUTION 1
        self.conv1 = nn.Conv2d(3, self.layer_output_channels[0], kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv1.weight, 0, 0.0001)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bnorm1 = nn.BatchNorm2d(self.layer_output_channels[0])

        # CONVOLUTION 2
        self.conv2 = nn.Conv2d(self.layer_output_channels[0],
                               self.layer_output_channels[1], kernel_size=5, stride=1, padding=2)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        self.relu2 = nn.ReLU(inplace=True)
        self.bnorm2 = nn.BatchNorm2d(self.layer_output_channels[1])
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2)

        # CONVOLUTION 3
        self.conv3 = nn.Conv2d(self.layer_output_channels[1],
                     self.layer_output_channels[2], kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv3.weight, 0, 0.04)
        self.relu3 = nn.ReLU(inplace=True)

        # CONVOLUTION 4
        self.conv4 = nn.Conv2d(self.layer_output_channels[2],
                               self.layer_output_channels[3], kernel_size=3, stride=1, padding=1)
        nn.init.normal_(self.conv4.weight, 0, 0.04)
        self.relu4 = nn.ReLU(inplace=True)

        # FULL CONNECTED
        self.classifier = nn.Linear(self.layer_output_channels[3] * 5 * 5, 10)
        nn.init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x

    def build_module(self, inhibition_layers: Dict[str, nn.Module]):
        """
        composes the module with optional specified inhibition layers as dictionary with 'inhib_{i}' as key,
         starting at i=1. Layers will be added after the i-th convolutional layer.

        :param inhibition_layers: a dictionary with inhibition layers and 'inhib_{i}' as key, starting with i=1

        """
        self.features.add_module("conv_1", self.conv1)
        if "inhib_1" in inhibition_layers.keys():
            self.features.add_module("inhib_1", inhibition_layers["inhib_1"])
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("bnorm_1", self.bnorm1)

        self.features.add_module("conv_2", self.conv2)
        if "inhib_2" in inhibition_layers.keys():
            self.features.add_module("inhib_2", inhibition_layers["inhib_2"])
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("bnorm_2", self.bnorm2)
        self.features.add_module("pool_2", self.pool2)

        self.features.add_module("conv_3", self.conv3)
        if "inhib_3" in inhibition_layers.keys():
            self.features.add_module("inhib_3", inhibition_layers["inhib_3"])
        self.features.add_module("relu_3", self.relu3)

        self.features.add_module("conv_4", self.conv4)
        if "inhib_4" in inhibition_layers.keys():
            self.features.add_module("inhib_4", inhibition_layers["inhib_4"])
        self.features.add_module("relu_4", self.relu4)

    def __str__(self):
        name = self.__class__.__name__
        if hasattr(self, 'freeze'):
            name += '_frozen' if self.freeze else ''
        return name


class Baseline(_AlexNetBase):
    """AlexNet Baseline for CIFAR-10"""

    def __init__(self):
        super().__init__()
        self.build_module({})


class BaselineCMap(_AlexNetBase):
    """AlexNet Baseline with Local Response Normalization reconstructed from
    https://code.google.com/archive/p/cuda-convnet/
    """

    def __init__(self):
        super().__init__()

        self.cnorm1 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)
        self.cnorm2 = nn.CrossMapLRN2d(9, k=2, alpha=10e-4, beta=0.75)

        self.features.add_module("conv_1", self.conv1)
        self.features.add_module("relu_1", self.relu1)
        self.features.add_module("pool_1", self.pool1)
        self.features.add_module("cnorm_1", self.cnorm1)
        self.features.add_module("bnorm_1", self.bnorm1)

        self.features.add_module("conv_2", self.conv2)
        self.features.add_module("relu_2", self.relu2)
        self.features.add_module("cnorm_2", self.cnorm2)
        self.features.add_module("bnorm_2", self.bnorm2)
        self.features.add_module("pool_2", self.pool2)

        self.features.add_module("conv_3", self.conv3)
        self.features.add_module("relu_3", self.relu3)

        self.features.add_module("conv_4", self.conv4)
        self.features.add_module("relu_4", self.relu4)


class AlexNetLC(_AlexNetBase):

    def __init__(self, widths: List[int] = None, damps: List[float] = None, strategy: str = None, optim: str = None,
                 self_connection: bool = False, pad: str = "circular"):
        super().__init__(widths, damps, strategy, optim, self_connection, pad)

        inhibition_layers = {}
        for i in range(1, self.coverage + 1):
            inhibition_layers.update({f"inhib_{i}": self.lateral_connect_layer_type(i,
                                                                                    self.layer_output_channels[i - 1])})

        self.build_module(inhibition_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


class SingleShotInhibitionNetwork(_AlexNetBase):
    """Baseline with added opportunity to add SSLC layers after convolutional layers"""

    def __init__(self, width: Union[int, List[int]], damp: Union[float, List], freeze: bool = True,
                 coverage: int = 1, self_connection=False, pad: str = "circular"):
        """

        :param self_connection:     whether the activated filter inhibits/excites itself or not
        :param pad:                 the padding, 'circular' or 'zeros'
        """
        super().__init__()

        # backward compatibility
        self.width = width if isinstance(width, List) else [width]
        self.damp = damp if isinstance(damp, List) else [damp]

        self.freeze = freeze
        self.coverage = coverage
        self.self_connection = self_connection
        self.pad = pad

        inhibition_layers = {}
        for i in range(1, coverage + 1):
            inhibition_layers.update(
                {f"inhib_{i}": SingleShotSemLC(ricker_width=self.width[i - 1], damp=self.damp[i - 1],
                                               learn_weights=not freeze, pad=pad, self_connection=self_connection)})

        self.build_module(inhibition_layers)


class ConvergedInhibitionNetwork(_AlexNetBase):
    """Baseline with added opportunity to add CLC layers after convolutional layers"""

    def __init__(self, width: Union[int, List[int]], damp: Union[float, List], freeze=True,
                 coverage: int = 1, self_connection=False, pad: str = "circular"):
        """

        :param self_connection:     whether the activated filter inhibits/excites itself or not
        :param pad:                 the padding, 'circular' or 'zeros'
        """
        super().__init__()

        # backward compatibility
        self.width = width if isinstance(width, List) else [width]
        self.damp = damp if isinstance(damp, List) else [damp]

        self.freeze = freeze
        self.coverage = coverage
        self.self_connection = self_connection
        self.pad = pad

        inhibition_layers = {}
        for i in range(1, coverage + 1):
            inhibition_layers.update({f"inhib_{i}":
                                          ConvergedSemLC(in_channels=self.layer_output_channels[i - 1],
                                                         ricker_width=self.width[i - 1], damp=self.damp[i - 1],
                                                         pad=pad, self_connection=self_connection) if not self.freeze else
                                          ConvergedFrozenSemLC(in_channels=self.layer_output_channels[i - 1],
                                                               ricker_width=self.width[i - 1],
                                                               damp=self.damp[i - 1], pad=pad,
                                                               self_connection=self_connection)})

        self.build_module(inhibition_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


class ParametricInhibitionNetwork(_AlexNetBase):
    """Baseline with added opportunity to add Paranetric CLC layers after convolutional layers"""

    def __init__(self, width: Union[int, List[int]], damp: Union[float, List], coverage: int = 1,
                 self_connection=False, pad: str = "circular"):
        """

        :param self_connection:     whether the activated filter inhibits/excites itself or not
        :param pad:                 the padding, 'circular' or 'zeros'
        """
        super().__init__()


        # backward compatibility
        self.width = width if isinstance(width, List) else [width]
        self.damp = damp if isinstance(damp, List) else [damp]
        self.coverage = coverage
        self.self_connection = self_connection
        self.pad = pad

        inhibition_layers = {}
        for i in range(1, coverage + 1):
            inhibition_layers.update(
                {f"inhib_{i}": ParametricSemLC(in_channels=self.layer_output_channels[i - 1],
                                               ricker_width=self.width[i - 1],
                                               initial_damp=self.damp[i - 1], pad=pad,
                                               self_connection=self_connection)})
            # TODO more general in_channels parameter

        self.build_module(inhibition_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.contiguous().view(x.size(0), 32 * 5 * 5)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    net_a = AlexNetLC([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1], strategy="CLC", optim="adaptive")
    net_b = AlexNetLC([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1], strategy="CLC-G", optim="frozen")
    net_c = Baseline()

    print(net_a.features)
    print(net_b.features)
    print(net_c.features)
