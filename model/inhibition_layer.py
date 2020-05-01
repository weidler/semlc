import torch
from torch import nn

from model.inhibition_module import InhibitionModule
from util import weight_initialization, ricker
from util.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero


# SINGLE SHOT

class SingleShotInhibition(InhibitionModule, nn.Module):
    """SSLC Layer.
    
    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

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

    @property
    def name(self):
        return f"SSLC {'Frozen' if not self.learn_weights else 'Adaptive'}"

    def _make_filter(self):
        return weight_initialization.mexican_hat(self.scope, damping=torch.tensor(self.damp, dtype=torch.float32),
                                                 width=torch.tensor(self.width, dtype=torch.float32),
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
        inhibition_filter = weight_initialization.mexican_hat(self.scope,
                                                              width=torch.tensor(ricker_width, dtype=torch.float32),
                                                              damping=torch.tensor(damp, dtype=torch.float32),
                                                              self_connect=self_connection)
        self.register_parameter("inhibition_filter", nn.Parameter(inhibition_filter, requires_grad=True))
        self.inhibition_filter.requires_grad = True

    @property
    def name(self):
        return f"CLC Adaptive"

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

    @property
    def name(self):
        return f"CLC Frozen"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.mexican_hat(self.scope, width=torch.tensor(self.width, dtype=torch.float32),
                                                 damping=torch.tensor(self.damp, dtype=torch.float32),
                                                 self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return convolve_3d_toeplitz(self.tpl_inv, activations)


# PARAMETRIC

class ParametricInhibition(InhibitionModule, nn.Module):
    """Inhibition layer using the single operation convergence point strategy with trainable parameters
    damping and width factor. Convergence point is determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

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

    @property
    def name(self):
        return f"CLC Parametric"

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


# GAUSSIAN FILTER

class SingleShotGaussian(SingleShotInhibition):
    def __init__(self, scope: int, width: int, damp: float, pad="circular", self_connection: bool = False):
        super().__init__(scope, width, damp, False, pad, self_connection)

    @property
    def name(self):
        return f"SSLC-G"

    def _make_filter(self):
        return weight_initialization.gaussian(self.scope, damping=self.damp, width=self.width,
                                              self_connect=self.self_connection)


class ConvergedGaussian(ConvergedFrozenInhibition):

    def __init__(self, scope: int, ricker_width: float, in_channels: int, damp: float = 0.12, pad="circular",
                 self_connection: bool = False):
        super().__init__(scope, ricker_width, in_channels, damp, pad, self_connection)

    @property
    def name(self):
        return f"CLC-G"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.gaussian(self.scope, width=self.width, damping=self.damp,
                                              self_connect=self.self_connection)
