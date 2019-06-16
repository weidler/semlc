import math
import random
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch import nn, optim
from torch.nn.functional import pad, mse_loss

from util import weight_initialization
from util.complex import div_complex
from util.linalg import toeplitz1d


def pad_roll(k, in_channels):
    """Zero-pad around filter, then roll to have center at i=0. Need to use concatenation to keep padding out of
    auto grad functionality. If torch's pad() function would be used, padding can be adjusted during optimization."""
    pad_left = torch.zeros((1, 1, (in_channels - scope) // 2), dtype=k.dtype)
    pad_right = torch.zeros((1, 1, (in_channels - scope) - pad_left.shape[-1]), dtype=k.dtype)
    return torch.cat((pad_left, k, pad_right), dim=-1).roll(math.floor(in_channels / 2) + 1)


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

    def __init__(self, scope: int, ricker_width: int, padding: str = "zeros", learn_weights: bool = False,
                 decay: float = 0.05, max_steps: int = 10, convergence_threshold: float = 0.00):
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
        self.W_rec.weight.data = weight_initialization.mexican_hat(scope, std=ricker_width, damping=0.12)
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

    def __init__(self, scope: int, ricker_width: int, in_channels: int):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels

        # inhibition filter, focused at i=0
        inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=0.12)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))

        # kronecker delta with mass at i=0 is identity to convolution with focus at i=0
        self.kronecker_delta = torch.zeros(in_channels).index_fill(0, torch.tensor([0]), 1)
        self.kronecker_delta = self.kronecker_delta.view((1, 1, 1, -1))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # bring the dimension that needs to be fourier transformed to the end
        activations = activations.permute((0, 2, 3, 1))

        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        kernel = pad_roll(self.inhibition_filter, self.in_channels)
        kernel = kernel.view((1, 1, 1, -1))

        # fourier transform
        fourier_activations = torch.rfft(activations, 1, onesided=False)
        fourier_filter = torch.rfft(self.kronecker_delta - kernel, 1, onesided=False)

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
        self.in_channels = in_channels

        # inhibition filter
        inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=0.12)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # pad roll the filter;
        # TODO inefficient to do this every time, but need to keep zeros out of autograd, better solutions?
        kernel = pad_roll(self.inhibition_filter, self.in_channels)

        # construct filter toeplitz
        tpl = toeplitz1d(kernel.squeeze(), self.in_channels)
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

    scope = 99
    depth = 100
    width = 14
    height = 14
    wavelet_width = 3

    tensor_in = torch.zeros((1, depth, width, height))
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[0, :, i, j] = torch.from_numpy(gaussian(depth, 4))

    inhibitor = Inhibition(scope, wavelet_width, padding="zeros", learn_weights=True)
    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, padding="zeros", learn_weights=True)
    inhibitor_conv = ConvergedInhibition(scope, wavelet_width, in_channels=depth)
    inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, in_channels=depth)

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].numpy(), label="Input")

    tensor_out = inhibitor(tensor_in)
    plt.plot(tensor_out[0, :, 4, 7].detach().numpy(), "-.", label="Single Shot")

    tensor_out_rec = inhibitor_rec(tensor_in)
    plt.plot(tensor_out_rec[0, :, 4, 7].detach().numpy(), label="Recurrent")

    tensor_out_conv = inhibitor_conv(tensor_in)
    plt.plot(tensor_out_conv[0, :, 4, 7].detach().numpy(), "--", label="Converged")

    tensor_out_tpl = inhibitor_tpl(tensor_in)
    plt.plot(tensor_out_tpl[0, :, 4, 7].detach().numpy(), ":", label="Converged Toeplitz")

    plt.legend()
    plt.show()

    # CHECK AUTO GRAD
    for test_layer in [inhibitor, inhibitor_rec, inhibitor_conv, inhibitor_tpl]:
        optimizer = optim.SGD(test_layer.parameters(), 0.01)
        start_time = time.time()
        for i in range(100):
            # print(f"BEFORE: {test_layer.inhibition_filter}")
            optimizer.zero_grad()

            out = test_layer(tensor_in)
            target = out * random.random()

            loss = mse_loss(out, target)
            loss.backward()

            optimizer.step()
            # print(f"AFTER: {test_layer.inhibition_filter}")
        print(f"Executed {test_layer.__class__.__name__} in {round(time.time() - start_time, 2)}s")

