import math

import matplotlib.pyplot as plt
import torch


def ricker(width, scope):
    A = 2 / (math.sqrt(3 * width) * (math.pi ** 0.25))
    start = -(scope - 1.0) / 2
    vec = [start + 1 * i for i in range(scope)]
    wavelet = [A * (math.exp(-j ** 2 / (2 * width ** 2))) * (1 - j ** 2 / width ** 2) for j in vec]
    ricked_tensor = torch.tensor(wavelet)
    return ricked_tensor


def dif_of_gauss(width, std, scope):
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
    dog = dog.tolist()
    return dog


if __name__ == "__main__":
    rickered = ricker(10, 20)
    doggy = dif_of_gauss(5, 10, 20)

    plt.plot(rickered, label="Ricker")
    plt.plot(doggy, label="DoG")

    plt.plot()