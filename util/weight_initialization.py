import torch
from scipy import signal


def mexican_hat(scope: int, std: float, damping: float=1) -> torch.Tensor:
    window = signal.ricker(scope, std) * damping
    distr_tensor = torch.tensor([[window]], dtype=torch.float)

    return distr_tensor


def gaussian(scope, std):
    window = signal.gaussian(scope, std=std)
    distr_tensor = torch.tensor([[window]], dtype=torch.float)

    return distr_tensor
