import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn

from model.deprecated_inhibition_layer import Conv3DSingleShotInhibition, Conv3DRecurrentInhibition
from model.inhibition_module import InhibitionModule
from util import weight_initialization
from util.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero


class ToeplitzSingleShotInhibition(nn.Module, InhibitionModule):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, ricker_width: int, damp: float, in_channels: int, learn_weights=False,
                 pad="circular",
                 analyzer=None):
        super().__init__()

        assert pad in ["circular", "zeros"]

        self.learn_weights = learn_weights
        self.in_channels = in_channels
        self.scope = scope
        self.analyzer = analyzer
        self.damp = damp
        self.is_circular = pad == "circular"

        inhibition_filter = weight_initialization.mexican_hat(scope, damping=damp, std=ricker_width)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))
        self.inhibition_filter.requires_grad = learn_weights

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, self.in_channels)

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl, activations)


class ConvergedToeplitzInhibition(nn.Module, InhibitionModule):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int, damp: float, in_channels: int, learn_weights: bool = True, pad="circular"):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"

        # inhibition filter
        inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter))
        self.inhibition_filter.requires_grad = learn_weights

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, self.in_channels)

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

    def __init__(self, scope: int, ricker_width: float, in_channels: int, damp: float = 0.12, pad="circular"):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"

        # inhibition filter
        self.inhibition_filter = weight_initialization.mexican_hat(scope, std=ricker_width, damping=damp)

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, self.in_channels)

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
    inhibitor = Conv3DSingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_ssi_tpl_zero = ToeplitzSingleShotInhibition(scope, wavelet_width, damp=damping, in_channels=depth,
                                                          pad="zeros",
                                                          learn_weights=True)
    inhibitor_ssi_tpl_circ = ToeplitzSingleShotInhibition(scope, wavelet_width, damp=damping, in_channels=depth,
                                                          pad="circular",
                                                          learn_weights=True)
    inhibitor_rec = Conv3DRecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_tpl_circ = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl_freeze_circ = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl_zero = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth, pad="zeros")
    inhibitor_tpl_freeze_zero = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth,
                                                                  pad="zeros")

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].cpu().numpy(), label="Input")

    tensor_out = inhibitor(tensor_in)
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), "-", label="Single Shot Conv3D Zeroed")

    tensor_out_tpl_ssi_zero = inhibitor_ssi_tpl_zero(tensor_in)
    plt.plot(tensor_out_tpl_ssi_zero[0, :, 4, 7].detach().cpu().numpy(), ".", label="Single Shot Tpl Zeroed")

    tensor_out_tpl_ssi_circ = inhibitor_ssi_tpl_circ(tensor_in)
    plt.plot(tensor_out_tpl_ssi_circ[0, :, 4, 7].detach().cpu().numpy(), "-.", label="Single Shot Tpl Circular")

    tensor_out_tpl_circ = inhibitor_tpl_circ(tensor_in)
    plt.plot(tensor_out_tpl_circ[0, :, 4, 7].detach().cpu().numpy(), "-", label="Converged Toeplitz Circular")

    tensor_out_tpl_freeze_circ = inhibitor_tpl_freeze_circ(tensor_in)
    plt.plot(tensor_out_tpl_freeze_circ[0, :, 4, 7].detach().cpu().numpy(), ".",
             label="Converged Toeplitz Frozen Circular")

    tensor_out_tpl_zero = inhibitor_tpl_zero(tensor_in)
    plt.plot(tensor_out_tpl_zero[0, :, 4, 7].detach().cpu().numpy(), "-", label="Converged Toeplitz Zeroed")

    tensor_out_tpl_freeze_zero = inhibitor_tpl_freeze_zero(tensor_in)
    plt.plot(tensor_out_tpl_freeze_zero[0, :, 4, 7].detach().cpu().numpy(), ".",
             label="Converged Toeplitz Frozen Zeroed")

    plt.title("Effects of Single Shot and Converged Inhibition for Different Padding Strategies")
    plt.legend()
    plt.show()
