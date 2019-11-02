import torch
from scipy import signal


def mexican_hat(scope: int, std: float, damping: float = 1, self_connect: bool = True) -> torch.Tensor:
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "WHAT?"

    hat = torch.tensor(signal.ricker(scope, std) * damping, dtype=torch.float)
    if not self_connect:
        hat[hat.shape[-1] // 2] = 0

    return hat


def gaussian(scope, std):
    window = signal.gaussian(scope, std=std)
    distr_tensor = torch.tensor(window, dtype=torch.float)

    return distr_tensor
