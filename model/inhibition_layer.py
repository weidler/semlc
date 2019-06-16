import math
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from scipy.linalg import toeplitz
from torch import nn
from torch.nn.functional import pad

from util import weight_initialization
from util.complex import div_complex
from util.linalg import toeplitz1d


class Inhibition(nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, ricker_width: int, padding: str = "zeros", learn_weights=False, analyzer=None):
        super().__init__()

        assert scope % 2 == 1
        assert padding in ["zeros", "cycle"]
        self.padding_strategy = padding
        self.scope = scope
        self.analyzer = analyzer

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
        self.convolver.weight.data = weight_initialization.mexican_hat(scope, damping=0.12, std=ricker_width)
        self.convolver.weight.data = self.convolver.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.convolver.parameters():
                param.requires_grad = False

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # catch weird scope-input-combinations; TODO do we really want this?
        if activations.shape[1] < self.scope:
            raise RuntimeError("Inhibition not possible. "
                               "Given activation has less filters than the Inhibitor's scope.")

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


class RecurrentInhibition(nn.Module):
    """Nice Inhibition Layer. """
    axs_convergence: List[Axes]
    fig_convergence: plt.Figure

    def __init__(self, scope: int, ricker_width: int, padding: str = "zeros", learn_weights: bool = False, decay: float = 0.05,
                 max_steps: int = 10, convergence_threshold: float = 0.00):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.max_steps = max_steps
        self.decay = decay
        self.padding_strategy = padding
        self.scope = scope
        self.convergence_threshold = convergence_threshold

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
        self.W_rec.weight.data = weight_initialization.mexican_hat(scope, std=ricker_width)
        self.W_rec.weight.data = self.W_rec.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.W_rec.parameters():
                param.requires_grad = False

        # figures for visualization
        self.fig_convergence, self.axs_convergence = plt.subplots(1, 2)
        self.fig_convergence.set_size_inches(12, 5)
        self.fig_convergence.suptitle("Convergence of Single Recursive Inhibition Process", )

    def forward(self, activations: torch.Tensor, plot_convergence=False) -> torch.Tensor:
        # catch weird scope-input-combinations; TODO do we really want this?
        if activations.shape[1] < self.scope:
            raise RuntimeError("Inhibition not possible. "
                               "Given activation has less filters than the Inhibitor's scope.")

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
        step_difference = math.inf
        step_differences = []
        converged_inhibition: torch.Tensor = activations.clone()

        while steps < self.max_steps and step_difference > self.convergence_threshold:
            inhib_rec = self.W_rec(converged_inhibition)
            phi = activations + inhib_rec

            previous_converged_inhibition = converged_inhibition
            converged_inhibition = (1 - self.decay) * converged_inhibition + self.decay * phi

            steps += 1
            step_difference = nn.functional.mse_loss(previous_converged_inhibition, converged_inhibition).item()
            step_differences.append(step_difference)

            if plot_convergence:
                self._plot_inhibition_convergence(activations, converged_inhibition, step_differences)

        # return inhibited activations without augmented channel dimension
        return converged_inhibition.squeeze_(dim=1)

    def _plot_inhibition_convergence(self, signal_in: torch.Tensor, signal_rec: torch.Tensor,
                                     step_differences: List[float]) -> None:

        self.axs_convergence[0].cla(), self.axs_convergence[1].cla()
        self.axs_convergence[0].set_ylim(top=10)
        self.axs_convergence[1].set_xlim(0, self.max_steps), self.axs_convergence[1].set_ylim(0, 0.03)

        x = list(range(self.scope))
        self.axs_convergence[0].plot(x, [f.item() for f in signal_in[0, 0, :, 0, 0]], label="Original")
        self.axs_convergence[0].plot(x, [f.item() for f in signal_rec[0, 0, :, 0, 0]], label="Recurrent")

        self.axs_convergence[1].plot(step_differences, label="Difference to last step")
        self.axs_convergence[1].plot([self.convergence_threshold for _ in list(range(self.max_steps))],
                                     label="Convergence Threshold", color="red")

        self.axs_convergence[0].legend(), self.axs_convergence[1].legend()

        plt.pause(0.001)


class ConvergedInhibition(nn.Module):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using deconvolution in the frequency domain with fourier transforms.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int):
        super().__init__()
        self.scope = scope

        # inhibition filter, focused at i=0
        self.inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=0.12).roll(math.ceil(scope / 2))
        self.inhibition_filter = self.inhibition_filter.view((1, 1, 1, -1))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(scope).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # bring the dimension that needs to be fourier transformed to the end
        m = activations.shape[1]
        activations = activations.permute((0, 2, 3, 1))

        # fourier transform
        fourier_activations = torch.rfft(activations, 1, onesided=False)
        pad_left = math.floor((m - self.scope) / 2)
        pad_right = (m - self.scope) - pad_left
        fourier_filter = torch.rfft(pad(self.kronecker_delta - self.inhibition_filter, [pad_left, pad_right]), 1, onesided=False)

        # divide in frequency domain, then bring back to time domain
        inhibited_tensor = torch.irfft(div_complex(fourier_activations, fourier_filter), 1, onesided=False)

        # restore original shape
        inhibited_tensor = inhibited_tensor.permute((0, 3, 1, 2))

        return inhibited_tensor


class ConvergedToeplitzInhibition(nn.Module):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int, in_channels: int):
        super().__init__()
        self.scope = scope

        # inhibition filter
        pad_left = (in_channels - scope) // 2
        pad_right = (in_channels - scope) - pad_left
        self.inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=0.12)
        self.inhibition_filter = pad(self.inhibition_filter, [pad_left, pad_right]).roll(-math.ceil(in_channels / 2))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        m = activations.shape[1]
        tpl = toeplitz1d(self.inhibition_filter.squeeze(), m)
        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # stack activation depth-columns for depth-wise convolution with tpl_inv
        stacked_activations = activations.unbind(dim=2)
        stacked_activations = torch.cat(stacked_activations, dim=2).permute((0, 2, 1))

        # convolve by multiplying with tpl
        convoluted = stacked_activations.matmul(tpl_inv)
        convoluted = convoluted.permute((0, 2, 1))

        # recover original shape
        return convoluted.view_as(activations)


if __name__ == "__main__":
    from scipy.signal import gaussian

    scope = 91
    depth = 101
    width = 14
    height = 14
    wavelet_width = 7
    tensor_in = torch.zeros((1, depth, width, height))
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[0, :, i, j] = torch.from_numpy(gaussian(depth, 4))

    inhibitor = Inhibition(scope, wavelet_width, padding="zeros")
    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, padding="zeros")
    inhibitor_conv = ConvergedInhibition(scope, wavelet_width)
    inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, in_channels=depth)

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].numpy(), label="Input")

    tensor_out = inhibitor(tensor_in)
    plt.plot(tensor_out[0, :, 4, 7].numpy(), "-.", label="Single Shot")

    # tensor_out_rec = inhibitor_rec(tensor_in)
    # plt.plot(tensor_out_rec[0, :, 4, 7].numpy(), label="Recurrent")

    tensor_out_conv = inhibitor_conv(tensor_in)
    plt.plot(tensor_out_conv[0, :, 4, 7].numpy(), "--", label="Converged")

    tensor_out_tpl = inhibitor_tpl(tensor_in)
    plt.plot(tensor_out_tpl[0, :, 4, 7].numpy(), ":", label="Converged Toeplitz")

    plt.legend()
    plt.show()
