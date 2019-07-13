import math
from typing import List

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib.axes import Axes
from torch import nn

from model.inhibition_module import InhibitionModule
from util import weight_initialization
from util.complex import div_complex
from util.linalg import toeplitz1d


def pad_roll(k, in_channels, scope):
    """Zero-pad around filter, then roll to have center at i=0. Need to use concatenation to keep padding out of
    auto grad functionality. If torch's pad() function would be used, padding can be adjusted during optimization."""
    pad_left = torch.zeros((1, 1, (in_channels - scope) // 2), dtype=k.dtype)
    pad_right = torch.zeros((1, 1, (in_channels - scope) - pad_left.shape[-1]), dtype=k.dtype)
    return torch.cat((pad_left, k, pad_right), dim=-1).roll(math.floor(in_channels / 2) + 1)


def convolve_3d_toeplitz(tpl_matrix: torch.Tensor, signal_tensor: torch.Tensor):
    # stack activation depth-columns for depth-wise convolution
    stacked_activations = signal_tensor.unbind(dim=2)
    stacked_activations = torch.cat(stacked_activations, dim=2).permute((0, 2, 1))

    # convolve by multiplying with tpl
    convolved_tensor = stacked_activations.matmul(tpl_matrix)
    convolved_tensor = convolved_tensor.permute((0, 2, 1))

    # recover original shape
    return convolved_tensor.view_as(signal_tensor)


def convolve_3d_fourier(filter: torch.Tensor, signal: torch.Tensor, delta: torch.Tensor):
    # bring the dimension that needs to be fourier transformed to the end
    signal = signal.permute((0, 2, 3, 1))

    # fourier transform
    fourier_activations = torch.rfft(signal, 1, onesided=False)
    fourier_filter = torch.rfft(delta - filter, 1, onesided=False)

    # divide in frequency domain, then bring back to time domain
    convolved_tensor = torch.irfft(div_complex(fourier_activations, fourier_filter), 1, onesided=False)

    # restore original shape
    convolved_tensor = convolved_tensor.permute((0, 3, 1, 2))

    return convolved_tensor


class SingleShotInhibition(nn.Module, InhibitionModule):
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
        self.convolver.weight.data = weight_initialization.mexican_hat(scope, damping=damp, std=ricker_width)
        self.convolver.weight.data = self.convolver.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.convolver.parameters():
                param.requires_grad = False

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


class ToeplitzSingleShotInhibition(nn.Module, InhibitionModule):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, ricker_width: int, damp: float, in_channels: int, learn_weights=False,
                 analyzer=None):
        super().__init__()

        self.learn_weights = learn_weights
        self.in_channels = in_channels
        self.scope = scope
        self.analyzer = analyzer
        self.damp = damp

        inhibition_filter = weight_initialization.mexican_hat(scope, damping=damp, std=ricker_width)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))
        self.inhibition_filter.requires_grad = learn_weights

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        kernel = pad_roll(self.inhibition_filter, self.in_channels, self.scope)

        # construct filter toeplitz
        tpl = toeplitz1d(kernel.squeeze(), self.in_channels)

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl, activations)


class RecurrentInhibition(nn.Module, InhibitionModule):
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
        self.W_rec.weight.data = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
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


class ConvergedInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: float, damp: float, in_channels: int, learn_weights: bool = True):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter, focused at i=0
        inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))
        self.inhibition_filter.requires_grad = learn_weights

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(in_channels).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        kernel = pad_roll(self.inhibition_filter, self.in_channels, self.scope)
        kernel = kernel.view((1, 1, 1, -1))

        return convolve_3d_fourier(kernel, activations, self.kronecker_delta)


class ConvergedFrozenInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms. Filter is frozen, implementation is optimized
    towards speed taking this into account.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int, in_channels: int, damp: float = 0.12):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter, focused at i=0
        self.inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
        self.inhibition_filter = pad_roll(self.inhibition_filter, self.in_channels, self.scope)
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


class ConvergedToeplitzInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int, damp: float, in_channels: int, learn_weights: bool = True):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter
        inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))
        self.inhibition_filter.requires_grad = learn_weights

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        kernel = pad_roll(self.inhibition_filter, self.in_channels, self.scope)

        # construct filter toeplitz
        tpl = toeplitz1d(kernel.squeeze(), self.in_channels)
        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl_inv, activations)


class ConvergedToeplitzFrozenInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: float, in_channels: int, damp: float = 0.12):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp

        # inhibition filter
        self.inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
        self.inhibition_filter = pad_roll(self.inhibition_filter, self.in_channels, self.scope)

        # construct filter toeplitz
        tpl = toeplitz1d(self.inhibition_filter.squeeze(), self.in_channels)
        self.tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return convolve_3d_toeplitz(self.tpl_inv, activations)


if __name__ == "__main__":
    from scipy.signal import gaussian

    # CUDA
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"USE CUDA: {use_cuda}.")

    # SETTINGS
    batches = 1
    scope = 99
    depth = 100
    width = 14
    height = 14
    wavelet_width = 6
    damping = 0.12

    tensor_in = torch.zeros((batches, depth, width, height))
    for b in range(batches):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(
                    gaussian(depth, 6)
                    + numpy.roll(gaussian(depth, 6), -(scope // 4)) * 0.5
                    + numpy.roll(gaussian(depth, 6), (scope // 4)) * 0.5
                    + numpy.roll(gaussian(depth, 6), -(scope // 2)) * 0.2
                    + numpy.roll(gaussian(depth, 6), (scope // 2)) * 0.2
                )

    simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)
    inhibitor = SingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_ssi_tpl = ToeplitzSingleShotInhibition(scope, wavelet_width, damp=damping, in_channels=depth,
                                                     learn_weights=True)
    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_conv = ConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_conv_freeze = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl_freeze = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].cpu().numpy(), label="Input")

    tensor_out = inhibitor(tensor_in)
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), "-.", label="Single Shot")

    tensor_out_tpl_ssi = inhibitor_ssi_tpl(tensor_in)
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), "--.", label="Single Shot Tpl")

    tensor_out_rec = inhibitor_rec(tensor_in)
    plt.plot(tensor_out_rec[0, :, 4, 7].detach().cpu().numpy(), label="Recurrent")

    tensor_out_conv = inhibitor_conv(tensor_in)
    plt.plot(tensor_out_conv[0, :, 4, 7].detach().cpu().numpy(), "--", label="Converged")

    tensor_out_conv_freeze = inhibitor_conv_freeze(tensor_in)
    plt.plot(tensor_out_conv_freeze[0, :, 4, 7].detach().cpu().numpy(), "--", label="Converged Frozen")

    tensor_out_tpl = inhibitor_tpl(tensor_in)
    plt.plot(tensor_out_tpl[0, :, 4, 7].detach().cpu().numpy(), ":", label="Converged Toeplitz")

    tensor_out_tpl_freeze = inhibitor_tpl_freeze(tensor_in)
    plt.plot(tensor_out_tpl_freeze[0, :, 4, 7].detach().cpu().numpy(), ":", label="Converged Toeplitz Frozen")

    plt.legend()
    plt.show()
