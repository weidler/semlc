import torch
from torch import nn

from core import weight_initialization
from core.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero
from layers.base import BaseSemLCLayer


class SemLC(BaseSemLCLayer):
    """Semantic lateral connectivity layers using the single operation convergence point strategy. Convergence point is
    determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float = 0.12,
                 pad="circular", self_connection: bool = False):
        super().__init__(hooked_conv, ricker_width, ricker_damp)
        self.damp = ricker_damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        self.lateral_filter = self._make_filter()

        # construct filter toeplitz
        if self.is_circular:
            self.tpl = toeplitz1d_circular(self.lateral_filter, self.in_channels)
        else:
            self.tpl = toeplitz1d_zero(self.lateral_filter, self.in_channels)

        tpl_inv = (self.tpl - torch.eye(*self.tpl.shape)).inverse()
        self.register_buffer("tpl_inv", tpl_inv)  # register so that its moved to correct devices

    @property
    def name(self):
        return f"SemLC"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.ricker_wavelet(self.in_channels - 1,
                                                    width=torch.tensor(self.width, dtype=torch.float32),
                                                    damping=torch.tensor(self.damp, dtype=torch.float32),
                                                    self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return ((-activations).permute(0, 2, 3, 1) @ self.tpl_inv.unsqueeze(0)).permute(0, 3, 1, 2).contiguous()


class SingleShotSemLC(BaseSemLCLayer):
    """One step Semantic lateral connectivity Layer.
    
    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float = 0.12, learn_weights=False,
                 pad="circular", self_connection: bool = False):
        super().__init__(hooked_conv, ricker_width, ricker_damp)

        assert pad in ["circular", "zeros"]

        self.learn_weights = learn_weights
        self.damp = ricker_damp
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        lateral_filter = self._make_filter()
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=learn_weights))
        self.lateral_filter.requires_grad = learn_weights

    @property
    def name(self):
        return f"SingleShot {'Frozen' if not self.learn_weights else 'Adaptive'}"

    def _make_filter(self):
        return weight_initialization.ricker_wavelet(self.in_channels - 1,
                                                    damping=torch.tensor(self.damp, dtype=torch.float32),
                                                    width=torch.tensor(self.width, dtype=torch.float32),
                                                    self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, activations.shape[1])

        # convolve by toeplitz
        return ((0 if self.self_connection else activations)
                + (activations.permute(0, 2, 3, 1) @ tpl.unsqueeze(0)).permute(0, 3, 1, 2).contiguous())


class AdaptiveSemLC(BaseSemLCLayer):
    """Semantic lateral connectivity layers using the single operation convergence point strategy. Convergence point is
    determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float = 0.12,
                 pad="circular", self_connection: bool = False):
        super().__init__(hooked_conv, ricker_width, ricker_damp)
        self.damp = ricker_damp
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection
        self.width = ricker_width

        # inhibition filter
        lateral_filter = weight_initialization.ricker_wavelet(self.in_channels - 1,
                                                              width=torch.tensor(ricker_width, dtype=torch.float32),
                                                              damping=torch.tensor(ricker_damp, dtype=torch.float32),
                                                              self_connect=self_connection)
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=True))
        self.lateral_filter.requires_grad = True

    @property
    def name(self):
        return f"Adaptive SemLC"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, activations.shape[1])
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, activations.shape[1])

        tpl_inv = (torch.eye(*tpl.shape, device=tpl.device) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(-tpl_inv, activations)


class ParametricSemLC(BaseSemLCLayer):
    """Semantic lateral connectivity layers using the single operation convergence point strategy with trainable parameters
    damping and width factor. Convergence point is determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float = 0.12,
                 pad="circular", self_connection: bool = False):
        super().__init__(hooked_conv, ricker_width, ricker_damp)

        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # parameters
        damp = torch.tensor(ricker_damp, dtype=torch.float32)
        width = torch.tensor(ricker_width, dtype=torch.float32)

        # inhibition filter
        self.register_parameter("damp", nn.Parameter(damp))
        self.register_parameter("width", nn.Parameter(width))
        self.damp.requires_grad, self.width.requires_grad = True, True

    @property
    def name(self):
        return f"Parametric SemLC"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # make filter from current damp and width
        self.lateral_filter = weight_initialization.ricker_wavelet(size=self.in_channels - 1, width=self.width,
                                                                   damping=self.damp, self_connect=self.self_connection)

        # construct filter toeplitz
        if self.is_circular:
            tpl = toeplitz1d_circular(self.lateral_filter, self.in_channels)
        else:
            tpl = toeplitz1d_zero(self.lateral_filter, self.in_channels)

        tpl_inv = (torch.eye(*tpl.shape, device=tpl.device) - tpl).inverse()

        # convolve by toeplitz
        return convolve_3d_toeplitz(tpl_inv, activations)


# COMPETITORS

class GaussianSemLC(SemLC):

    def __init__(self, hooked_conv: nn.Conv2d, ricker_width: float, ricker_damp: float = 0.12,
                 pad="circular", self_connection: bool = False):
        super().__init__(hooked_conv, ricker_width, ricker_damp, pad, self_connection)

    @property
    def name(self):
        return f"Gaussian SemLC"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.matching_gaussian(self.in_channels - 1, width=self.width, ricker_damping=self.damp,
                                                       self_connect=self.self_connection)


class LRN(BaseSemLCLayer):

    def __init__(self, hooked_conv, ricker_width: float = 0, ricker_damp: float = 0):
        super().__init__(hooked_conv, ricker_width, ricker_damp)

        self.wrapped_lrn = nn.LocalResponseNorm(size=9, k=2, alpha=10e-4, beta=0.75)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.wrapped_lrn(activations)


class CMapLRN(BaseSemLCLayer):

    def __init__(self, hooked_conv, ricker_width: float = 0, ricker_damp: float = 0):
        super().__init__(hooked_conv, ricker_width, ricker_damp)

        self.wrapped_lrn = nn.CrossMapLRN2d(size=9, k=2, alpha=10e-4, beta=0.75)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.wrapped_lrn(activations)