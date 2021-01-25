import torch
from torch import nn

from layers.base import BaseSemLCLayer


class LRN(BaseSemLCLayer):

    def __init__(self, hooked_conv, ricker_width: float = 0, ricker_damp: float = 0):
        super().__init__(hooked_conv, ricker_width, ricker_damp)

        self.wrapped_lrn = nn.CrossMapLRN2d(size=9, k=2, alpha=10e-4, beta=0.75)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        return self.wrapped_lrn(activations)