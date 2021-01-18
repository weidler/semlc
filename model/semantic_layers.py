import torch
from torch import nn

from model.inhibition_module import BaseSemLC
from util import weight_initialization, ricker
from util.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero


# SINGLE SHOT

class SingleShotSemLC(BaseSemLC):
    """One step Semantic lateral connectivity Layer.
    
    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, in_channels: int, ricker_width: float, damp: float, learn_weights=False, pad="circular",
                 self_connection: bool = False):
        super().__init__()

        assert pad in ["circular", "zeros"]

        self.learn_weights = learn_weights
        self.in_channels = in_channels
        self.damp = damp
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        lateral_filter = self._make_filter()
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=learn_weights))
        self.lateral_filter.requires_grad = learn_weights

    @property
    def name(self):
        return f"SSLC {'Frozen' if not self.learn_weights else 'Adaptive'}"

    def _make_filter(self):
        return weight_initialization.mexican_hat(self.in_channels - 1, damping=torch.tensor(self.damp, dtype=torch.float32),
                                                 width=torch.tensor(self.width, dtype=torch.float32),
                                                 self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, activations.shape[1])

        # convolve by toeplitz
        return (0 if self.self_connection else activations) + convolve_3d_toeplitz(tpl, activations)


# CONVERGED

class ConvergedSemLC(BaseSemLC):
    """Semantic lateral connectivity layer using the single operation convergence point strategy. Convergence point is
    determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, in_channels: int, ricker_width: int, damp: float, pad="circular", self_connection: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        lateral_filter = weight_initialization.mexican_hat(self.in_channels - 1,
                                                              width=torch.tensor(ricker_width, dtype=torch.float32),
                                                              damping=torch.tensor(damp, dtype=torch.float32),
                                                              self_connect=self_connection)
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=True))
        self.lateral_filter.requires_grad = True

    @property
    def name(self):
        return f"CLC Adaptive"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, activations.shape[1])

        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(-tpl_inv, activations)


class ConvergedFrozenSemLC(BaseSemLC):
    """Semantic lateral connectivity layer using the single operation convergence point strategy. Convergence point is determined
    using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, in_channels: int, ricker_width: float, damp: float = 0.12, pad="circular",
                 self_connection: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.damp = damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        self.lateral_filter = self._make_filter()

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, self.in_channels)

        self.tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

    @property
    def name(self):
        return f"CLC Frozen"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.mexican_hat(self.in_channels - 1, width=torch.tensor(self.width, dtype=torch.float32),
                                                 damping=torch.tensor(self.damp, dtype=torch.float32),
                                                 self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return convolve_3d_toeplitz(-self.tpl_inv, activations)


# PARAMETRIC

class ParametricSemLC(BaseSemLC):
    """Semantic lateral connectivity layer using the single operation convergence point strategy with trainable parameters
    damping and width factor. Convergence point is determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, in_channels: int, ricker_width: float, initial_damp: float, pad="circular",
                 self_connection: bool = False):
        super().__init__()
        self.in_channels = in_channels

        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # parameters
        damp = torch.tensor(initial_damp, dtype=torch.float32)
        width = torch.tensor(ricker_width, dtype=torch.float32)

        # inhibition filter
        self.register_parameter("damp", nn.Parameter(damp))
        self.register_parameter("width", nn.Parameter(width))
        self.damp.requires_grad, self.width.requires_grad = True, True

    @property
    def name(self):
        return f"CLC Parametric"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # make filter from current damp and width
        self.lateral_filter = ricker.ricker(scope=self.in_channels - 1, width=self.width, damp=self.damp,
                                               self_connect=self.self_connection)

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, self.in_channels)

        tpl_inv = (torch.eye(*tpl.shape) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl_inv, activations)


# GAUSSIAN FILTER

class ConvergedGaussianSemLC(ConvergedFrozenSemLC):

    def __init__(self, in_channels: int, ricker_width: float, damp: float = 0.12, pad="circular",
                 self_connection: bool = False):
        super().__init__(in_channels, ricker_width, damp, pad, self_connection)

    @property
    def name(self):
        return f"CLC-G"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.gaussian(self.in_channels - 1, width=self.width, damping=self.damp,
                                              self_connect=self.self_connection)
