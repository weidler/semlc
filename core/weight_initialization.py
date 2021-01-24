"""Functions providing initialization of lateral connectivity filters."""
import math
import timeit
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from scipy import signal
from torch import nn

from utilities.util import closest_factors


def ricker_scipy(size: int, width: float, damping: float = 1, self_connect: bool = True) -> torch.Tensor:
    """Compose a Ricker wavelet filter using scipy (not differentiable).

    :param size:                size of the output vector
    :param width:               width of the wavelet
    :param damping:             damping factor scaling the amplitude of the wavelet
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    assert size % 2 != 0, "Scope must have an odd number of dimensions."
    assert size > 0, "WHAT?"

    hat = torch.tensor(signal.ricker(size, width) * damping, dtype=torch.float)
    if not self_connect:
        hat[hat.shape[-1] // 2] = 0

    return hat


def ricker_wavelet(size: int, width: torch.Tensor, damping: torch.Tensor, self_connect: bool = True):
    """Compose a differentiable Ricker wavelet filter.

    :param size:                the size
    :param width:               width of the wavelet
    :param damping:             damping factor scaling the amplitude of the wavelet
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    assert size % 2 != 0, "Scope must have an odd number of dimensions."
    assert size > 0, "Scope must be positive"

    a = damping * (2 / (torch.sqrt(3 * width) * (math.pi ** 0.25)))
    start = -(size - 1.0) / 2
    vec = torch.arange(start, start + size)

    # pre-calculations
    vec_squared = torch.square(vec)
    width_squared = torch.square(width)

    mod = (1 - vec_squared / width_squared)
    gauss = torch.exp(-vec_squared / (2 * width_squared))

    wavelet = a * mod * gauss

    if not self_connect:
        wavelet[wavelet.shape[-1] // 2] = 0

    return wavelet


def gaussian(size: int, width: float, damping: float, self_connect: bool = True):
    """Compose a Gaussian filter.

    :param size:                size of the output vector
    :param width:               width of the gaussian
    :param damping:             damping factor scaling the amplitude of the gaussian
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    assert size % 2 != 0, "Scope must have an odd number of dimensions."
    assert size > 0, "WHAT?"

    start = -(size - 1.0) / 2
    x = torch.tensor([start + i for i in range(size)])
    gaussian_filter = damping * torch.exp(-(torch.pow(x - 0, 2) / (2 * (width * width))))
    if not self_connect:
        gaussian_filter[gaussian_filter.shape[-1] // 2] = 0

    return gaussian_filter


def matching_gaussian(size: int, width: float, ricker_damping: float, self_connect: bool = True):
    base_gaussian = gaussian(size, width, 1, self_connect=self_connect)
    ricker_center_value = ricker_damping * 2 / (math.sqrt(3 * width) * (math.pi ** 0.25))

    return base_gaussian * ricker_center_value


# GABOR FILTERS

def gabor_filter(size, theta, lamb, sigma, gamma):
    """Generate complex Gabor kernel as a torch tensor.

    Args:
        size:   (tuple or int)  specifies the size of the bounding box of the kernel
        theta:  (float)         orientation of the filter in radians
        lamb:   (float)         wavelength of the carrier, in pixels; size / lamb gives the number of peaks
        sigma:  (float)         std of the Gaussian envelope
        gamma:  (float)         ellipticity of the Gaussian envelope

    Returns:

    """
    radius = (int(size[0] / 2.0), int(size[1] / 2.0))
    x, y = torch.meshgrid([torch.arange(-radius[0], radius[0] + 1),
                           torch.arange(-radius[1], radius[1] + 1)])
    theta = torch.tensor(theta, dtype=torch.float32)
    xprime = x * torch.cos(theta) + y * torch.sin(theta)
    yprime = -x * torch.sin(theta) + y * torch.cos(theta)

    envelope = torch.exp(- (xprime ** 2 + gamma ** 2 * yprime ** 2) / (2 * sigma ** 2))
    carrier = torch.exp((2 * math.pi * (xprime / lamb)) * 1j)

    complex_gabor = carrier * envelope

    return complex_gabor


def generate_gabor_filter_bank(size: Tuple[int, ...], lamb, n_filters: int = 8, part="complex", scale: bool = False):
    """Generate a bank of Gabor filters."""
    assert part in ["complex", "real", "imag"]

    if scale:
        cfs = closest_factors(n_filters)
        n_thetas, n_scales = max(cfs), min(cfs)
    else:
        n_thetas, n_scales = n_filters, 1

    thetas = torch.arange(0, math.pi, math.pi / n_thetas)
    scales = torch.arange(1, 2, 1 / n_scales)

    filter_bank = []
    for theta in thetas:
        for scale in scales:
            g = gabor_filter(size=size, theta=theta, lamb=lamb * scale, sigma=size[0] / 12 * scale, gamma=0.6)
            if part == "real":
                g = g.real
            elif part == "imag":
                g = g.imag

            filter_bank.append(g)

    return filter_bank


def fix_layer_weights_to_gabor(layer, scale=True):
    """Fix the weights of a convolutional or complex cell convolutional layer to gabor filters.

    TODO handle RGB correctly."""
    lambdas = layer.kernel_size[0] / 4

    gabor = generate_gabor_filter_bank(size=tuple(layer.kernel_size), lamb=lambdas,
                                       n_filters=layer.out_channels, part="real", scale=scale)

    with torch.no_grad():
        layer.weight = nn.Parameter(torch.stack(gabor, dim=0)
                                    .unsqueeze(1)
                                    .to(layer.weight.device)
                                    .expand(-1, layer.in_channels, -1, -1),
                                    requires_grad=False)


if __name__ == "__main__":
    scope = 47
    width = 3
    damping = 0.1
    self_connect = True

    # EFFICIENCY
    print(f"{timeit.timeit(lambda: ricker_scipy(scope, width, damping), number=1000)} (scipy)")
    print(
        f"{timeit.timeit(lambda: ricker_wavelet(scope, torch.tensor(width, dtype=torch.float32), torch.tensor(damping, dtype=torch.float32)), number=1000)} (ours)")

    # VISUAL INSPECTION
    gauss = gaussian(scope, width, damping, self_connect=self_connect)
    mgauss = matching_gaussian(scope, width, damping, self_connect=self_connect)
    mh = ricker_wavelet(scope, torch.tensor(width, dtype=torch.float32), torch.tensor(damping, dtype=torch.float32),
                        self_connect=self_connect)

    start = -(scope - 1.0) / 2
    x = torch.tensor([start + i for i in range(scope)])

    plt.axvline(0, color="grey", ls="--")
    plt.axhline(0, color="grey", ls="--")
    plt.plot(x, gauss.detach().cpu().tolist(), label="Gaussian")
    plt.plot(x, mgauss.detach().cpu().tolist(), label="Matched Gaussian")
    plt.plot(x, mh.detach().cpu().tolist(), label="Mexican hat")

    plt.legend()
    plt.show()
