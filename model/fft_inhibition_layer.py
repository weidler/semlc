#!/usr/bin/env python
"""Implementations of inhibition layers using fourier transform. Turned out to be a lost slower to backpropagate,
hence this is sort of deprecated but kept here for speed demonstration."""
from torch import nn
import torch

from util.convolution import pad_roll, convolve_3d_fourier
from model.inhibition_module import InhibitionModule
from util import weight_initialization
from util.complex import div_complex


class FFTConvergedInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    @property
    def name(self):
        return "Converged Adaptive (FFT)"

    def __init__(self, scope: int, ricker_width: float, damp: float, in_channels: int, learn_weights: bool = True):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter, focused at i=0
        inhibition_filter = weight_initialization.mexican_hat(scope, width=ricker_width, damping=damp, self_connect=False)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter, requires_grad=learn_weights))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(in_channels).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        # cannot think of anything better, will probably do for now, see how it performs
        kernel = pad_roll(self.inhibition_filter.view(1, 1, -1), self.in_channels, self.scope)
        kernel = kernel.view((1, 1, 1, -1))

        return convolve_3d_fourier(kernel, activations, self.kronecker_delta)


class FFTConvergedFrozenInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms. Filter is frozen, implementation is optimized
    towards speed taking this into account.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    @property
    def name(self):
        return "Converged Frozen (FFT)"

    def __init__(self, scope: int, ricker_width: int, in_channels: int, damp: float = 0.12):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter, focused at i=0
        self.inhibition_filter = weight_initialization.mexican_hat(scope, width=ricker_width, damping=damp,
                                                                   self_connect=False)
        self.inhibition_filter = pad_roll(self.inhibition_filter.view(1, 1, -1), self.in_channels, self.scope)
        self.inhibition_filter = self.inhibition_filter.view((1, 1, 1, -1))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(in_channels).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

        # filter in frequency domain
        self.fourier_filter = torch.rfft(self.kronecker_delta - self.inhibition_filter, 1, onesided=False)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # bring the dimension that needs to be fourier transformed to the end
        activations = activations.permute((0, 2, 3, 1))

        # divide in frequency domain, then bring back to time domain
        fourier_activations = torch.rfft(activations, 1, onesided=False)
        inhibited_tensor = torch.irfft(div_complex(fourier_activations, self.fourier_filter), 1, onesided=False)

        return inhibited_tensor.permute((0, 3, 1, 2))
