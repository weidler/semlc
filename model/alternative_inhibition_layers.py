#!/usr/bin/env python
"""Implementations of inhibition layers that are deprecated approaches."""
import math
from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import nn, __init__

from model.inhibition_module import InhibitionModule
from util import weight_initialization

import torch


class Conv3DSingleShotInhibition(nn.Module, InhibitionModule):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, ricker_width: int, damp: float, padding: str = "zeros", learn_weights=False,
                 analyzer=None):
        super().__init__()

        assert scope % 2 == 1
        assert padding in ["zeros", "cycle"]
        self.padding_strategy = padding
        self.scope = scope
        self.analyzer = analyzer
        self.damp = damp

        self.convolver: nn.Conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(scope, 1, 1),
            stride=(1, 1, 1),
            padding=(scope // 2, 0, 0) if padding == "zeros" else (0, 0, 0),
            dilation=(1, 1, 1),
            bias=0
        )

        # apply gaussian
        self.convolver.weight.data = weight_initialization.mexican_hat(scope, width=ricker_width, damping=damp)
        self.convolver.weight.data = self.convolver.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.convolver.parameters():
                param.requires_grad = False

    @property
    def name(self):
        return f"SingleShot (Conv3D) {'Frozen' if not self.learn_weights else 'Adaptive'}"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:

        # augment channel dimension
        activations = activations.unsqueeze(dim=1)

        # apply cycle padding strategy if necessary
        if self.padding_strategy == "cycle":
            activations = torch.cat((
                activations[:, :, -self.scope // 2 + 1:, :, :],
                activations,
                activations[:, :, :self.scope // 2, :, :]), dim=2)

        # inhibit
        inhibitions = self.convolver(activations)

        # analyse
        if self.analyzer is not None:
            self.analyzer.visualize(activations, inhibitions)

        # return inhibited activations without augmented channel dimension
        return inhibitions.squeeze_(dim=1)


@DeprecationWarning
class Conv3DRecurrentInhibition(nn.Module, InhibitionModule):
    """Nice Inhibition Layer. """
    axs_convergence: List[Axes]
    fig_convergence: plt.Figure

    def __init__(self, scope: int, ricker_width: int, damp: float = 0.12, padding: str = "zeros",
                 learn_weights: bool = False,
                 decay: float = 0.05, max_steps: int = 10, convergence_threshold: float = 0.00):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.max_steps = max_steps
        self.decay = decay
        self.padding_strategy = padding
        self.scope = scope
        self.convergence_threshold = convergence_threshold
        self.damp = damp

        # recurrent convolution filter
        self.W_rec: nn.Conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(scope, 1, 1),
            stride=(1, 1, 1),
            padding=(scope // 2, 0, 0) if padding == "zeros" else (0, 0, 0),
            dilation=(1, 1, 1),
            bias=0
        )

        # apply gaussian
        self.W_rec.weight.data = weight_initialization.mexican_hat(scope, width=ricker_width, damping=damp)
        self.W_rec.weight.data = self.W_rec.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.W_rec.parameters():
                param.requires_grad = False

    def forward(self, activations: torch.Tensor, plot_convergence=False) -> torch.Tensor:
        # augment channel dimension
        activations = activations.unsqueeze(dim=1)

        # apply cycle padding strategy if necessary
        if self.padding_strategy == "cycle":
            activations = torch.cat((
                activations[:, :, -self.scope // 2 + 1:, :, :],
                activations,
                activations[:, :, :self.scope // 2, :, :]), dim=2)

        # inhibit
        steps = 0
        dt = 1
        step_difference = math.inf
        converged_inhibition: torch.Tensor = activations.clone()

        while steps < self.max_steps and step_difference > self.convergence_threshold:
            inhib_rec = self.W_rec(converged_inhibition)
            phi = activations + inhib_rec

            converged_inhibition = dt * phi
            steps += 1

        # return inhibited activations without augmented channel dimension
        return converged_inhibition.squeeze_(dim=1)


def pad_shift_zero(k: torch.Tensor, in_channels, scope):
    """Shift and Zero-pad filter to the right such that filter's center is at i=0 but convolutional padding is zeros not
    circular padding."""
    # pad_left = torch.zeros((1, 1, (in_channels - scope) // 2), dtype=k.dtype)
    # pad_right = torch.zeros((1, 1, (in_channels - scope) - pad_left.shape[-1]), dtype=k.dtype)
    # return torch.cat((pad_left, k, pad_right), dim=-1).roll(math.floor(in_channels / 2) + 1)
    assert len(k.shape) == 3

    shifted = k[:, :, math.floor(k.shape[2] / 2):]
    zeros = torch.zeros((1, 1, in_channels - shifted.shape[2]), dtype=k.dtype)
    return torch.cat((shifted, zeros), dim=2)