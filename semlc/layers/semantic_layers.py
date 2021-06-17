from typing import Tuple

import torch
from torch import nn

from core import weight_initialization
from core.convolution import toeplitz1d_circular, convolve_3d_toeplitz, toeplitz1d_zero
from layers import BaseSemLCLayer


class SemLC(BaseSemLCLayer):
    """Semantic lateral connectivity layers using the single operation convergence point group. Convergence point is
    determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float = 2, damping: float = 0.12,
                 pad="circular", self_connection: bool = False, rings: int = 1):
        super().__init__(hooked_conv, widths, ratio, damping, rings=rings)
        self.ricker_damp = damping
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # inhibition filter
        self.lateral_filter = self._make_filter()

        # construct filter toeplitz
        if self.is_circular:
            self.tpl = toeplitz1d_circular(self.lateral_filter, self.ring_size)
        else:
            self.tpl = toeplitz1d_zero(self.lateral_filter, self.rring_size)

        tpl_inv = (self.tpl - torch.eye(*self.tpl.shape)).inverse()
        self.register_buffer("tpl_inv", tpl_inv)  # register so that its moved to correct devices

    @property
    def name(self):
        return f"SemLC"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.difference_of_gaussians(self.ring_size - 1,
                                                             widths=(torch.tensor(self.widths[0], dtype=torch.float32),
                                                                     torch.tensor(self.widths[1], dtype=torch.float32)),
                                                             ratio=torch.tensor(self.ratio),
                                                             damping=torch.tensor(self.ricker_damp,
                                                                                  dtype=torch.float32),
                                                             self_connect=self.self_connection)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # todo parallel without loop?
        per_ring_output = []
        for ring in activations.split(self.ring_size, dim=1):
            per_ring_output.append(((-ring).permute(0, 2, 3, 1) @ self.tpl_inv.unsqueeze(0)).permute(0, 3, 1, 2).contiguous())

        stacked = torch.cat(per_ring_output, dim=1)
        return stacked


class SingleShotSemLC(BaseSemLCLayer):
    """One step Semantic lateral connectivity Layer.
    
    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float = 2, damping: float = 0.12,
                 learn_weights=False,
                 pad="circular", self_connection: bool = False, rings: int = 1):
        super().__init__(hooked_conv, widths, ratio, damping, rings=rings)

        assert pad in ["circular", "zeros"]

        self.learn_weights = learn_weights
        self.ricker_damp = damping
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        lateral_filter = self._make_filter()
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=learn_weights))
        self.lateral_filter.requires_grad = learn_weights

    @property
    def name(self):
        return f"SingleShot {'Frozen' if not self.learn_weights else 'Adaptive'}"

    def _make_filter(self):
        return weight_initialization.difference_of_gaussians(self.ring_size - 1,
                                                             damping=torch.tensor(self.ricker_damp,
                                                                                  dtype=torch.float32),
                                                             widths=(torch.tensor(self.widths[0], dtype=torch.float32),
                                                                     torch.tensor(self.widths[1], dtype=torch.float32)),
                                                             ratio=torch.tensor(self.ratio),
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
    """Semantic lateral connectivity layers using the single operation convergence point group. Convergence point is
    determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float = 2, damping: float = 0.12,
                 pad="circular", self_connection: bool = False, rings: int = 1):
        super().__init__(hooked_conv, widths, ratio, damping, rings=rings)
        self.ricker_damp = damping
        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # inhibition filter
        lateral_filter = weight_initialization.difference_of_gaussians(self.ring_size - 1,
                                                                       widths=(torch.tensor(self.widths[0],
                                                                                            dtype=torch.float32),
                                                                               torch.tensor(self.widths[1],
                                                                                            dtype=torch.float32)),
                                                                       ratio=torch.tensor(self.ratio),
                                                                       damping=torch.tensor(damping,
                                                                                            dtype=torch.float32),
                                                                       self_connect=self_connection)
        self.register_parameter("lateral_filter", nn.Parameter(lateral_filter, requires_grad=True))
        self.lateral_filter.requires_grad = True

    @property
    def name(self):
        return f"SemLC-A"

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
    """Semantic lateral connectivity layers using the single operation convergence point group with trainable parameters
    damping and width factor. Convergence point is determined using the inverse of a Toeplitz matrix.

    Input shape:
        N x C x H x W
        --> where N is the number of batches, C the number of filters_per_group, and H and W are spatial dimensions.
    """

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float = 2, damping: float = 0.12,
                 pad="circular", self_connection: bool = False, rings: int = 1):
        super().__init__(hooked_conv, widths, ratio, damping, rings=rings)

        assert pad in ["circular", "zeros"]
        self.is_circular = pad == "circular"
        self.self_connection = self_connection

        # parameters
        damp = torch.tensor(self.damping, dtype=torch.float32)
        width = torch.tensor(self.widths, dtype=torch.float32)

        # inhibition filter
        self.register_parameter("damp", nn.Parameter(damp))
        self.register_parameter("width_epsps", nn.Parameter(width[0]))
        self.register_parameter("width_ipsps", nn.Parameter(width[1]))
        self.damp.requires_grad, self.width_epsps.requires_grad, self.width_ipsps.requires_grad = True, True, True

    @property
    def name(self):
        return f"SemLC-P"

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # make filter from current damp and width
        self.lateral_filter = weight_initialization.difference_of_gaussians(size=self.ring_size - 1,
                                                                            widths=(self.width_epsps, self.width_ipsps),
                                                                            ratio=torch.tensor(self.ratio),
                                                                            damping=self.damp,
                                                                            self_connect=self.self_connection)

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

    def __init__(self, hooked_conv: nn.Conv2d, widths: Tuple[float, float], ratio: float = 2, damping: float = 0.12,
                 pad="circular", self_connection: bool = False, rings: int = 1):
        super().__init__(hooked_conv, widths, ratio, damping, pad, self_connection, rings=rings)

    @property
    def name(self):
        return f"SemLC-G"

    def _make_filter(self) -> torch.Tensor:
        return weight_initialization.gaussian(self.ring_size - 1,
                                              width=torch.tensor(self.widths[0]),
                                              damping=torch.tensor(self.damping),
                                              self_connect=self.self_connection)


class LRN(BaseSemLCLayer):

    def __init__(self, hooked_conv, widths: Tuple[float, float] = (0, 0), ratio: float = 2, damping: float = 0):
        super().__init__(hooked_conv, widths, ratio, damping)

        self.wrapped_lrn = nn.LocalResponseNorm(size=9, k=2, alpha=10e-4, beta=0.75)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.wrapped_lrn(activations)


class CMapLRN(BaseSemLCLayer):

    def __init__(self, hooked_conv, widths: Tuple[float, float] = (0, 0), ratio: float = 2, damping: float = 0):
        super().__init__(hooked_conv, widths, ratio, damping)

        self.wrapped_lrn = nn.CrossMapLRN2d(size=9, k=2, alpha=10e-4, beta=0.75)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.wrapped_lrn(activations)


class LRNSemLCChain(BaseSemLCLayer):
    def __init__(self, hooked_conv, widths: Tuple[float, float] = (0, 0), ratio: float = 2, damping: float = 0):
        super().__init__(hooked_conv, widths, ratio, damping)

        self.lrn = LRN(hooked_conv)
        self.semlc = SemLC(hooked_conv, widths, damping)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.semlc(self.lrn(activations))


class SemLCLRNChain(LRNSemLCChain):

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.lrn(self.semlc(activations))
