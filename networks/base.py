from typing import Tuple, Union

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from config import RGB_TO_GREYSCALE_WEIGHTS
from core.weight_initialization import fix_layer_weights_to_gabor
from layers.base import BaseSemLCLayer
from utilities.util import closest_factors


class BaseNetwork(nn.Module):

    def __init__(self, input_shape: Tuple[int, int, int], lateral_layer_function: BaseSemLCLayer, lateral_before=True):
        super().__init__()

        input_shape = tuple(input_shape)
        assert len(input_shape) == 3, f"Got {len(input_shape)} domains in input shape, but expected 3 (C, H, W)."

        self.lateral_before = lateral_before
        self.input_shape = input_shape
        self.input_channels, self.input_height, self.input_width = input_shape

        self.lateral_layer_function = lateral_layer_function

        # quick check attributes
        self.is_grayscale = self.input_channels == 1
        self.is_lateral = self.lateral_layer_function is not None
        self.lateral_type = self.lateral_layer_function.func.__name__ if self.is_lateral else None

    def generate_random_input(self, batch_size=1):
        return torch.randn((batch_size,) + self.input_shape)

    def init_gabors(self):
        if hasattr(self, "conv_one"):
            fix_layer_weights_to_gabor(self.conv_one)
        else:
            raise NotImplementedError("Cannot find conv_one layer in model and as such cannot init to gabor filters.")

    # LOGGING AND INSPECTION
    def serialize_meta(self):
        return {
            "network_type": self.__class__.__name__,
            "input_channels": self.input_channels,
            "input_width": self.input_width,
            "input_height": self.input_height,
            "is_lateral": self.is_lateral,
            "lateral_type": self.lateral_type,
        }

    def visualize_v1_filters(self, channel=None, shown_filters: torch.Tensor = None, ignored_ids=None):
        conv_one_name = None
        if ignored_ids is None:
            ignored_ids = []

        if shown_filters is None:
            shown_filters = torch.arange(self.conv_one.out_channels)

        for potential_name in ["conv_one", "conv1", "conv_1"]:
            if hasattr(self, potential_name):
                conv_one_name = potential_name
                break

        if conv_one_name is None:
            raise AttributeError("Network has no V1 layer conforming to naming convention.")

        v1_layer = self.__getattr__(conv_one_name)
        rgb_to_greyscale_factor = torch.ones((1, 3, 1, 1))
        rgb_to_greyscale_factor[0, :, 0, 0] = torch.tensor(RGB_TO_GREYSCALE_WEIGHTS)

        if not self.is_complex:
            if channel is None:
                filters = (v1_layer.weight.clone() * rgb_to_greyscale_factor).mean(dim=1)
            else:
                filters = v1_layer.weight.clone()[:, channel, ...]
        else:
            if channel is None:
                real_filters = (v1_layer.real_kernels.weight.clone() * rgb_to_greyscale_factor).mean(dim=1)
                imaginary_filters = (v1_layer.imaginary_kernels.weight.clone() * rgb_to_greyscale_factor).mean(dim=1)
            else:
                real_filters = v1_layer.real_kernels.weight.clone()[:, channel, ...]
                imaginary_filters = v1_layer.imaginary_kernels.weight.clone()[:, channel, ...]

            filters = torch.cat((real_filters, imaginary_filters), dim=-1)

        imgs = [f.numpy() for f in filters[..., shown_filters, :, :].detach().unbind(-3)]
        rs, cs = closest_factors(len(imgs) - len(ignored_ids))
        fig, axs = plt.subplots(
            nrows=rs,
            ncols=cs,
        )
        fig.set_size_inches(cs * 3, rs * 4)
        i = 0
        for row in axs:
            for col in row:
                col: Axes
                col.imshow(imgs[i], cmap=plt.gray(),
                           **(dict(vmin=None[0], vmax=None[1]) if None is not None else dict()))
                col.tick_params(
                    axis='both',
                    which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelbottom=False,
                    labeltop=False,
                    labelleft=False,
                    labelright=False,
                )

                if self.is_complex:
                    col.spines['bottom'].set_color('red')
                    col.spines['top'].set_color('red')
                    col.spines['right'].set_color('red')
                    col.spines['left'].set_color('red')

                    col.set_title(str(i + 1), fontdict=dict(fontsize=20))

                    col.axvline(imgs[i].shape[-1] / 2 - 0.5, color="red")

                i += 1
                while i in ignored_ids:
                    i += 1

                if len(imgs) - 1 < i:
                    break
            if len(imgs) - 1 < i:
                break

    # OPTIMIZATION BUILDERS
    def make_preferred_optimizer(self) -> Optimizer:
        return optim.Adam(self.parameters(), lr=0.001)

    @staticmethod
    def make_preferred_lr_schedule(optimizer) -> Union[_LRScheduler, lr_scheduler.ReduceLROnPlateau]:
        return lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200])

    @staticmethod
    def make_preferred_criterion():
        return nn.CrossEntropyLoss()
