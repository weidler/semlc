"""Functions providing initialization of lateral connectivity filters_per_group."""
import itertools
import math
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import torch
from scipy import signal
from torch import nn

from utilities.util import closest_factors


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

    width = width.float()
    damping = damping.float()

    a = damping * (2 / (torch.sqrt(3 * width) * (math.pi ** 0.25)))
    start = -(size - 1.0) / 2
    vec = torch.arange(start, start + size, device=width.device)

    # pre-calculations
    vec_squared = torch.square(vec)
    width_squared = torch.square(width)

    mod = (1 - vec_squared / width_squared)
    gauss = torch.exp(-vec_squared / (2 * width_squared))

    wavelet = a * mod * gauss

    if not self_connect:
        wavelet[wavelet.shape[-1] // 2] = 0

    return wavelet


def difference_of_gaussians(size: int,
                            widths: Tuple[torch.Tensor, torch.Tensor],
                            ratio: torch.Tensor,
                            damping: torch.Tensor,
                            self_connect: bool = True):
    """Difference of Gaussians kernel.

    Peak is normalized to height 1."""
    excitation_gaussian = gaussian(size, widths[0], torch.tensor(1) * ratio, True)
    inhibition_gaussian = gaussian(size, widths[1], torch.tensor(1), True)

    dog = (excitation_gaussian - inhibition_gaussian) / (ratio - 1 + 1e-8) * damping

    if not self_connect:
        dog[dog.shape[-1] // 2] = 0

    return dog


def gaussian(size: int, width: torch.Tensor, damping: torch.Tensor, self_connect: bool = True):
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
    x = torch.tensor([start + i for i in range(size)], device=width.device)
    gaussian_filter = damping * torch.exp(-(torch.pow(x, 2) / (2 * (width * width))))
    if not self_connect:
        gaussian_filter[gaussian_filter.shape[-1] // 2] = 0

    return gaussian_filter


def normalized_gaussian(size: int, width: torch.Tensor, damping: torch.Tensor, self_connect: bool = True):
    """Compose a normalized Gaussian filter.

    :param size:                size of the output vector
    :param width:               width of the gaussian
    :param damping:             damping factor scaling the amplitude of the gaussian
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    return gaussian(size, width, damping, self_connect) * (1 / (2 * math.pi * (width ** 2)))


def matching_gaussian(size: int, width: float, ricker_damping: float, self_connect: bool = True):
    base_gaussian = gaussian(size, width, 1, self_connect=self_connect)
    ricker_center_value = ricker_damping * 2 / (math.sqrt(3 * width) * (math.pi ** 0.25))

    return base_gaussian * ricker_center_value


# difference of beta

def beta(size: int, alphas: torch.Tensor, betas: torch.Tensor, damping: torch.Tensor, self_connect: bool = True):
    """Probability density of the Beta distribution."""

    # start = -(size - 1.0) / 2
    samples = torch.tensor([0 + i for i in range(size)], dtype=torch.float32) / size

    top = torch.pow(samples, torch.subtract(alphas, 1.)) * torch.pow(torch.subtract(torch.tensor(1.), samples),
                                                                     torch.subtract(betas, 1.))
    bab = torch.multiply(torch.exp(torch.lgamma(alphas)),
                         torch.exp(torch.lgamma(betas)) / torch.exp(torch.lgamma(torch.add(alphas, betas))))

    out = top / bab

    return out


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


def generate_gabor_filter_bank(size: Tuple[int, ...], lamb, n_filters: int = 8, part="complex", scale: bool = False)\
        -> List[torch.Tensor]:
    """Generate a bank of Gabor filters_per_group."""
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
            g = gabor_filter(size=size, theta=theta, lamb=lamb * scale, sigma=size[0] / 6 * scale, gamma=1)
            if part == "real":
                g = g.real
            elif part == "imag":
                g = g.imag

            filter_bank.append(g)

    return filter_bank


def fix_layer_weights_to_gabor(layer, scale=True):
    """Fix the weights of a convolutional or complex cell convolutional layer to gabor filters_per_group.

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
    scope = 27
    width = torch.tensor(3.)
    damping = torch.tensor(1)
    self_connect = True

    # for w in [0.1, 1,2,3,4,5,6,7]:
    #     plt.plot(normalized_gaussian(scope, w, damping), label=str(w))
    #
    # plt.legend()
    # plt.show()

    # total_psps = []
    # for wa, wb in itertools.combinations([3, 3], 2):
    #     a = dog_mexican_hat(scope, (torch.tensor(wa), torch.tensor(wb)), (damping), self_connect)
    #     plt.bar(list(range(scope)), a, label=f"Total PSP [{wa}, {wb}]")
    #     plt.legend()
    #     plt.show()
    #
    #     total_psps.append((a, f"{wa} - {wb}"))
    #
    # for tpsp, name in total_psps:
    #     plt.plot(tpsp, label=name)

    do = False
    if do:
        r_range = range(2, 10)
        w_range = range(4, 14)

        fig, axs = plt.subplots(len(r_range), len(w_range))

        i = 0
        for r in r_range:
            j = 0
            for w in w_range:
                the_dog = difference_of_gaussians(scope, (width, torch.tensor(w)), torch.tensor(r), damping, self_connect)
                axs[i][j].bar(range(scope), the_dog)
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                # plt.plot(ricker_wavelet(scope, width, damping), label="Ricker")
                j += 1
            i += 1

        plt.legend()
        plt.show()

    do = True
    if do:
        w1_range = range(1, 5)
        w2_range = range(1, 8)

        fig, axs = plt.subplots(len(w1_range), len(w2_range))
        fig.set_size_inches((24, 12))

        i = 0
        for w1 in w1_range:
            j = 0
            for w2 in w2_range:
                the_dog = difference_of_gaussians(scope, (torch.tensor(w1), torch.tensor(w1 + w2)), torch.tensor(2), damping, self_connect)
                axs[i][j].bar(range(scope), the_dog, label=f"{w1}, {w1+w2}")
                axs[i][j].set_xticks([])
                axs[i][j].set_yticks([])
                axs[i][j].legend()
                # plt.plot(ricker_wavelet(scope, width, damping), label="Ricker")
                j += 1
            i += 1

        plt.show()

    # i = 4
    # b = beta(scope, torch.tensor(float(i)), torch.tensor(float(i)), torch.tensor(1.), self_connect)
    # a = beta(scope, torch.tensor(float(i + 6)), torch.tensor(float(i + 4)), torch.tensor(1.), self_connect)
    # plt.plot(a - b, label=str(i))


