import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn

from model.inhibition_module import InhibitionModule
from util import weight_initialization, ricker
from util.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero


# SINGLE SHOT

class SingleShotInhibition(InhibitionModule, nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, ricker_width: float, damp: float, learn_weights=False, pad="circular",
                 self_connection: bool = False):
        super().__init__()

        assert pad in ["circular", "zeros"]

        self.learn_weights = learn_weights
        self.scope = scope
        self.damp = damp
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        inhibition_filter = self._make_filter()
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter, requires_grad=learn_weights))
        self.inhibition_filter.requires_grad = learn_weights

    def _make_filter(self):
        return weight_initialization.mexican_hat(self.scope, damping=self.damp, width=self.width,
                                                 self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, activations.shape[1])

        # convolve by toeplitz
        return (0 if self.self_connection else activations) + convolve_3d_toeplitz(tpl, activations)


# CONVERGED

class ConvergedInhibition(InhibitionModule, nn.Module):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: int, damp: float, pad="circular",
                 self_connection: bool = False):
        super().__init__()
        super()
        self.scope = scope
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        inhibition_filter = weight_initialization.mexican_hat(self.scope, width=ricker_width, damping=damp,
                                                              self_connect=self_connection)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter, requires_grad=True))
        self.inhibition_filter.requires_grad = True

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, activations.shape[1])

        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl_inv, activations)


class ConvergedFrozenInhibition(InhibitionModule, nn.Module):
    """Inhibition layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, scope: int, ricker_width: float, in_channels: int, damp: float = 0.12, pad="circular",
                 self_connection: bool = False):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        self.inhibition_filter = self._make_filter()

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, self.in_channels)

        self.tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.mexican_hat(self.scope, width=self.width, damping=self.damp,
                                                 self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return convolve_3d_toeplitz(self.tpl_inv, activations)


# GAUSSIAN FILTER

class SingleShotGaussianChannelFilter(SingleShotInhibition):
    def __init__(self, scope: int, width: int, damp: float, pad="circular", self_connection: bool = False):
        super().__init__(scope, width, damp, False, pad, self_connection)

    def _make_filter(self):
        return weight_initialization.gaussian(self.scope, damping=self.damp, width=self.width,
                                              self_connect=self.self_connection)


class ConvergedGaussianChannelFilter(ConvergedFrozenInhibition):

    def __init__(self, scope: int, ricker_width: float, in_channels: int, damp: float = 0.12, pad="circular",
                 self_connection: bool = False):
        super().__init__(scope, ricker_width, in_channels, damp, pad, self_connection)

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.gaussian(self.scope, width=self.width, damping=self.damp,
                                              self_connect=self.self_connection)


class RecurrentInhibition(SingleShotGaussianChannelFilter):

    def __init__(self, scope: int, width: float, damp: float, pad="circular",
                 self_connection: bool = False, filter_distribution: str = "ricker"):
        self.filter_distribution = filter_distribution
        super().__init__(scope, width, damp, pad, self_connection)

    def _make_filter(self):
        if self.filter_distribution == "ricker":
            return weight_initialization.mexican_hat(self.scope, damping=self.damp, width=self.width,
                                                     self_connect=self.self_connection)
        elif self.filter_distribution == "gaussian":
            return weight_initialization.gaussian(self.scope, width=self.width, damping=self.damp,
                                                  self_connect=self.self_connection)
        else:
            raise NotImplementedError("Unknown inhibition filter distribution")

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, activations.shape[1])

        # convolve by toeplitz
        inhibited_activations = activations.clone()
        for _ in range(100):
            # x = inhibited_activations[0, :, 0, 0].detach().cpu().numpy()
            # plt.cla()
            # plt.plot(x)
            # plt.pause(0.1)
            inhibited_activations = activations + convolve_3d_toeplitz(tpl, inhibited_activations)

        return inhibited_activations


# PARAMETRIC

class ParametricInhibition(InhibitionModule, nn.Module):

    def __init__(self, scope: int, initial_ricker_width: float, initial_damp: float, in_channels: int,
                 pad="circular", self_connection: bool = False):
        super().__init__()
        self.scope = scope
        self.in_channels = in_channels
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # parameters
        damp = torch.tensor(initial_damp, dtype=torch.float32)
        width = torch.tensor(initial_ricker_width, dtype=torch.float32)

        # inhibition filter
        self.register_parameter("damp", nn.Parameter(damp))
        self.register_parameter("width", nn.Parameter(width))
        self.damp.requires_grad, self.width.requires_grad = True, True

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # make filter from current damp and width
        self.inhibition_filter = ricker.ricker(scope=self.scope, width=self.width, damp=self.damp,
                                          self_connect=self.self_connection)

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.inhibition_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.inhibition_filter, self.in_channels)

        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl_inv, activations)


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
    self_connect = False

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
    inhibitor_ssi_tpl_zero = SingleShotInhibition(scope, wavelet_width, damp=damping, learn_weights=True, pad="zeros",
                                                  self_connection=self_connect)
    inhibitor_ssi_tpl_circ = SingleShotInhibition(scope, wavelet_width, damp=damping, learn_weights=True,
                                                  pad="circular", self_connection=self_connect)
    inhibitor_gaussian_ssi = SingleShotGaussianChannelFilter(scope, wavelet_width, damp=damping, pad="circular",
                                                             self_connection=self_connect)
    inhibitor_tpl_circ = ConvergedInhibition(scope, wavelet_width, damp=damping,
                                             self_connection=self_connect)
    inhibitor_tpl_freeze_circ = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth,
                                                          self_connection=self_connect)
    inhibitor_tpl_zero = ConvergedInhibition(scope, wavelet_width, damp=damping, pad="zeros",
                                             self_connection=self_connect)
    inhibitor_tpl_freeze_zero = ConvergedFrozenInhibition(scope, wavelet_width, in_channels=depth, damp=damping,
                                                          pad="zeros", self_connection=self_connect)

    inhibitor_gaussian = ConvergedGaussianChannelFilter(scope, wavelet_width, damp=damping,
                                                        self_connection=self_connect, in_channels=depth)

    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, self_connection=self_connect)

    inhibitor_gaussian_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, self_connection=self_connect)

    inhibitor_parametrized = ParametricInhibition(scope, wavelet_width, initial_damp=damping,
                                                  self_connection=self_connect, in_channels=depth)

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].cpu().numpy(), label="Input")

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

    tensor_out_parametrized = inhibitor_parametrized(tensor_in)
    plt.plot(tensor_out_parametrized[0, :, 4, 7].detach().cpu().numpy(), "--",
             label="Parametrized Toeplitz Circular")

    tensor_out_rec = inhibitor_rec(tensor_in)
    plt.plot(tensor_out_rec[0, :, 4, 7].detach().cpu().numpy(), "--",
             label="Recurrent Inhibition")

    # GAUSSIAN WAVELETS
    tensor_out_gaussian = inhibitor_gaussian(tensor_in)
    plt.plot(tensor_out_gaussian[0, :, 4, 7].detach().cpu().numpy(), ".",
             label="Converged Gaussian Channel Filter")

    # tensor_out_gaussian_rec = inhibitor_gaussian_rec(tensor_in)
    # plt.plot(tensor_out_gaussian_rec[0, :, 4, 7].detach().cpu().numpy(), "--",
    #          label="Recurrent Gaussian Channel Filter")
    #
    # tensor_out_gaussian_ssi = inhibitor_gaussian_ssi(tensor_in)
    # plt.plot(tensor_out_gaussian_ssi[0, :, 4, 7].detach().cpu().numpy(), "--",
    #          label="Single Shot Gaussian Channel Filter")

    plt.title(f"Effects of Different Inhibition Strategies ")
    plt.legend()
    plt.savefig(f"../documentation/figures/strategy_effects.pdf", format="pdf")
    plt.show()
