#!/usr/bin/env python
"""Implementations of inhibition layers using fourier transform. Turned out to be a lot slower to backpropagate,
hence this is sort of deprecated but kept here for speed demonstration."""
from torch import nn
import torch

from core import pad_roll, convolve_3d_fourier
from layers import BaseSemLCLayer
from core import weight_initialization
from utilities.complex import div_complex


class FFTAdaptiveSemLC(BaseSemLCLayer):
    """Inhibition layers using the single operation convergence point group. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    @property
    def name(self):
        return f"SemLC-A (FFT)"

    def __init__(self, hooked_conv: nn.Conv2d, widths: float, damping: float = 0.12):
        super().__init__(hooked_conv, widths, 2, damping)

        # inhibition filter, focused at i=0
        inhibition_filter = weight_initialization.ricker_wavelet(self.in_channels - 1,
                                                                 width=torch.tensor(self.widths),
                                                                 damping=torch.tensor(self.damping),
                                                                 self_connect=False)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter, requires_grad=True))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(self.in_channels).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        # cannot think of anything better, will probably do for now, see how it performs
        kernel = pad_roll(self.inhibition_filter.view(1, 1, -1), self.in_channels, self.in_channels - 1)
        kernel = kernel.view((1, 1, 1, -1))

        return convolve_3d_fourier(kernel, activations, self.kronecker_delta)


class FFTSemLC(BaseSemLCLayer):
    """Inhibition layers using the single operation convergence point group. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms. Filter is frozen, implementation is optimized
    towards speed taking this into account.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    @property
    def name(self):
        return "SemLC (FFT)"

    def __init__(self, hooked_conv: nn.Conv2d, widths: float, damping: float = 0.12):
        super().__init__(hooked_conv, widths, 2, damping)  # inhibition filter, focused at i=0
        self.inhibition_filter = weight_initialization.ricker_wavelet(self.in_channels - 1,
                                                                      width=torch.tensor(self.widths),
                                                                      damping=torch.tensor(self.damping),
                                                                      self_connect=False)
        self.inhibition_filter = pad_roll(self.inhibition_filter.view(1, 1, -1), self.in_channels, self.in_channels - 1)
        self.inhibition_filter = self.inhibition_filter.view((1, 1, 1, -1))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(self.in_channels).index_fill(0, torch.tensor([0]), 1)
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
