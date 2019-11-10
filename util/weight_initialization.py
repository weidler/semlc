import torch
from scipy import signal

import matplotlib.pyplot as plt


def mexican_hat(scope: int, width: float, damping: float = 1, self_connect: bool = True) -> torch.Tensor:
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "WHAT?"

    hat = torch.tensor(signal.ricker(scope, width) * damping, dtype=torch.float)
    if not self_connect:
        hat[hat.shape[-1] // 2] = 0

    return hat


def gaussian(scope: int, width: float, damping: float, self_connect: bool = True):
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "WHAT?"

    gaussian_filter = damping * torch.tensor(signal.gaussian(scope, std=width), dtype=torch.float)
    if not self_connect:
        gaussian_filter[gaussian_filter.shape[-1] // 2] = 0

    return gaussian_filter


if __name__ == "__main__":
    gauss = gaussian(27, 3, 0.1)
    plt.plot(gauss.detach().cpu().tolist())
    plt.show()
