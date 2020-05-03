"""Functions providing initialization of lateral connectivity filters."""
import timeit

import torch
from scipy import signal
import math

import matplotlib.pyplot as plt

from util.convolution import toeplitz1d_circular


def mexican_hat_scipy(scope: int, width: float, damping: float = 1, self_connect: bool = True) -> torch.Tensor:
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "WHAT?"

    hat = torch.tensor(signal.ricker(scope, width) * damping, dtype=torch.float)
    if not self_connect:
        hat[hat.shape[-1] // 2] = 0

    return hat


def mexican_hat(scope: int, width: torch.Tensor, damping: torch.Tensor, self_connect: bool = True):
    """Composes a ricker wavelet filter.

    :param width:               the width of the wavelet
    :param damping:                the damping factor
    :param scope:               the scope
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "Scope must be positive"

    if not isinstance(width, torch.Tensor) or isinstance(damping, torch.Tensor):
        width = torch.tensor(width, dtype=torch.float32)
        damping = torch.tensor(damping, dtype=torch.float32)

    a = 2 / (torch.sqrt(3 * width) * (math.pi ** 0.25))
    start = -(scope - 1.0) / 2
    vec = torch.tensor([start + i for i in range(scope)])
    wavelet = damping * a * (torch.exp(-vec ** 2 / (2 * width ** 2))) * (1 - vec ** 2 / width ** 2)

    if not self_connect:
        wavelet[wavelet.shape[-1] // 2] = 0

    return wavelet


def gaussian(scope: int, width: float, damping: float, self_connect: bool = True):
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "WHAT?"

    start = -(scope - 1.0) / 2
    x = torch.tensor([start + i for i in range(scope)])
    gaussian_filter = damping * torch.exp(-(torch.pow(x - 0, 2) / (2 * (width * width))))
    if not self_connect:
        gaussian_filter[gaussian_filter.shape[-1] // 2] = 0

    return gaussian_filter


def matching_gaussian(scope: int, width: float, ricker_damping: float, self_connect: bool = True):
    base_gaussian = gaussian(scope, width, 1, self_connect=self_connect)
    ricker_center_value = ricker_damping * 2 / (math.sqrt(3 * width) * (math.pi ** 0.25))

    return base_gaussian * ricker_center_value


def stabilize_profile(profile: torch.Tensor, signal_size):
    """Stabilize the profile by guaranteeing all eigenvalues to be strictly negative."""
    tpl_matrix = toeplitz1d_circular(profile, signal_size)
    eigenvalues = torch.eig(tpl_matrix * torch.ones(signal_size) - torch.eye(signal_size))[0]

    print(eigenvalues)

    return profile


def is_stable_profile(profile: torch.Tensor, signal_size):
    """Check if a profile is stable by test whether largest eigenvalue of (J = Uz - 1) is negative."""
    tpl_matrix = toeplitz1d_circular(profile, signal_size)
    jacobian = tpl_matrix * torch.ones(signal_size) * 10 - torch.eye(signal_size)
    eigenvalues = torch.eig(jacobian)[0]

    return torch.max(eigenvalues[:, 0]) < 0


def dif_of_gauss(scope, width, std):
    """Realize the mexican hat wavelet as a difference of gaussians.

    :param width:       the width of the wavelet
    :param std:         the standard deviation
    :param scope:       the scope

    :return:            the wavelet
    """
    start = -(scope - 1.0) / 2
    stdb = std * 2
    vec = [start + 1 * i for i in range(scope)]
    gaus1 = torch.tensor(
        [((1 / (std * (math.sqrt(2 * math.pi)))) * (math.e ** -(((j - width) / std) ** 2) / 2)) for j in vec])
    gaus2 = torch.tensor(
        [((1 / (stdb * (math.sqrt(2 * math.pi)))) * (math.e ** -(((j - width) / stdb) ** 2) / 2)) for j in vec])

    dog = torch.sub(gaus1, gaus2)

    return dog


if __name__ == "__main__":
    scope = 47
    width = 3
    damping = 0.1
    self_connect = True

    # EFFICIENCY
    print(f"{timeit.timeit(lambda: mexican_hat_scipy(scope, width, damping), number=1000)} (scipy)")
    print(f"{timeit.timeit(lambda: mexican_hat(scope, torch.tensor(width, dtype=torch.float32), torch.tensor(damping, dtype=torch.float32)), number=1000)} (ours)")

    # VISUAL INSPECTION
    gauss = gaussian(scope, width, damping, self_connect=self_connect)
    mgauss = matching_gaussian(scope, width, damping, self_connect=self_connect)
    mh = mexican_hat(scope, torch.tensor(width, dtype=torch.float32), torch.tensor(damping, dtype=torch.float32),
                     self_connect=self_connect)

    print(is_stable_profile(gauss, 99))
    print(is_stable_profile(mgauss, 99))
    print(is_stable_profile(mh, 99))

    start = -(scope - 1.0) / 2
    x = torch.tensor([start + i for i in range(scope)])

    plt.axvline(0, color="grey", ls="--")
    plt.axhline(0, color="grey", ls="--")
    plt.plot(x, gauss.detach().cpu().tolist(), label="Gaussian")
    plt.plot(x, mgauss.detach().cpu().tolist(), label="Matched Gaussian")
    plt.plot(x, mh.detach().cpu().tolist(), label="Mexican hat")

    plt.legend()
    plt.show()
