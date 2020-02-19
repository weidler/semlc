import math

import matplotlib.pyplot as plt
import torch


def ricker(width: torch.Tensor, damp: torch.Tensor, scope: int, self_connect: bool = True):
    """
    composes the ricker wavelet

    :param width:               the width of the wavelet
    :param damp:                the damping factor
    :param scope:               the scope
    :param self_connect:        whether to form a connection of a neuron to itself

    :return:                    the wavelet
    """
    assert scope % 2 != 0, "Scope must have an odd number of dimensions."
    assert scope > 0, "Scope must be positive"
    assert isinstance(width, torch.Tensor) and isinstance(damp, torch.Tensor), "Width and Damp must be tensors"

    A = 2 / (torch.sqrt(3 * width) * (math.pi ** 0.25))
    start = -(scope - 1.0) / 2
    vec = torch.tensor([start + 1 * i for i in range(scope)])
    wavelet = damp * A * (torch.exp(-vec ** 2 / (2 * width ** 2))) * (1 - vec ** 2 / width ** 2)

    if not self_connect:
        wavelet[wavelet.shape[-1] // 2] = 0

    return wavelet


def dif_of_gauss(width, std, scope):
    """
    realize the wavelet as a difference of gaussians

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
    # gaus1 = torch.tensor([(math.e ** -(((j - width)/ (std))**2)/2) for j in vec])
    # gaus2 = torch.tensor([(math.e ** -(((j - width)/ (stdb))**2)/2) for j in vec])
    gaus2 = torch.tensor(
        [((1 / (stdb * (math.sqrt(2 * math.pi)))) * (math.e ** -(((j - width) / stdb) ** 2) / 2)) for j in vec])
    # dog = [(((1/(scope*(math.sqrt(2*math.pi))))*(math.e**-((j-width)**2)/(2*scope**2))))-(((1/(scopeb*(math.sqrt(2*math.pi))))*(math.e**-((j-width)**2)/(2*scopeb**2)))) for j in vec]
    dog = torch.sub(gaus1, gaus2)
    return dog


if __name__ == "__main__":
    from util.weight_initialization import mexican_hat

    scope = 27
    width = 3
    damping = 0.1

    mh = mexican_hat(scope, width, damping)
    rickered = ricker(torch.tensor(width, dtype=torch.float32), torch.tensor(damping, dtype=torch.float32), scope=scope)
    dog = dif_of_gauss(width=width, std=width, scope=scope)

    plt.plot(mh.detach().cpu().numpy(), label="SciPy Ricker")
    plt.plot(rickered.detach().cpu().numpy(), "--", label="Ricker")
    plt.plot(dog.detach().cpu().numpy(), label="DoG")
    plt.legend()
    plt.axhline(color="black", lw=1)

    plt.show()
