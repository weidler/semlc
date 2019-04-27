import torch
from scipy import signal


def mexican_hat(scope: int, std: float) -> torch.Tensor:
    window = signal.ricker(scope, std)
    distr_tensor = torch.Tensor([[window]])

    return distr_tensor


def gaussian(scope, std):
    window = signal.gaussian(scope, std=std)
    distr_tensor = torch.Tensor([[window]])

    return distr_tensor
