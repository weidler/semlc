from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from model.network.alexnet_paper import InhibitionNetwork, Baseline
from util.filter_ordering import mse


def visualize_filters(filters: List[Tensor]):
    """
    visualizes the given filters, rows * cols must be >= number of filters
    :param filters: the filters
    :param rows: number of rows in plot
    :param cols: number of cols in plot
    """
    rows, cols = get_dim_for_plot(len(filters))
    fig, axs = plt.subplots(rows, cols)
    f = 0
    for row in range(rows):
        for col in range(cols):
            img = np.swapaxes(filters[f], 0, 2)
            img = np.ndarray.astype(np.interp(img, (img.min(), img.max()), (0, 1)), dtype=float)
            axs[row, col].imshow(img)
            f += 1
    plt.show()


def plot_unsorted_and_sorted_filters(filters: List[Tensor], sorted_filters: List[Tensor]):
    """
    visualizes the given filters, rows * cols must be >= number of filters
    :param filters: the filters
    :param sorted_filters: the sorted filters
    :param rows: number of rows in plot
    :param cols: number of cols in plot
    """
    rows, cols = get_dim_for_plot(len(filters))
    fig, axs = plt.subplots(rows, cols)
    fig.suptitle('unsorted')
    fig2, axs2 = plt.subplots(rows, cols)
    fig2.suptitle('sorted')
    f = 0
    for row in range(rows):
        for col in range(cols):
            img = np.swapaxes(filters[f], 0, 2)
            img = np.ndarray.astype(np.interp(img, (img.min(), img.max()), (0, 1)), dtype=float)
            img2 = np.swapaxes(sorted_filters[f], 0, 2)
            img2 = np.ndarray.astype(np.interp(img2, (img2.min(), img2.max()), (0, 1)), dtype=float)
            axs[row, col].imshow(img)
            axs2[row, col].imshow(img2)
            f += 1

    plt.show()


def show_ordering_difference(filters: List[Tensor], sorted_filters: List[Tensor]):
    """
    prints the previous ordering followed by the ordering after sorting in tuples
    :param filters: the filters
    :param sorted_filters: the sorted filters
    """
    for i in range(len(filters)):
        for j in range(len(filters)):
            if np.all(filters[i] == sorted_filters[j]):
                print(i + 1, j + 1)


def get_dim_for_plot(n):
    """
    returns the "best" dimension for plotting filters, e.g. n = 6 returns 2,3 and 16 returns 4,4
    :param n: the number of filters
    :return: a tuple of rows and cols for the plot
    """
    nsqrt = np.math.ceil(np.math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n / val)
        if val2 * val == float(n):
            solution = True
        else:
            val -= 1
    if val == 1:
        return get_dim_for_plot(n + 1)
    if val <= val2:
        return val, val2
    else:
        return val2, val


if __name__ == "__main__":
    # how to use
    strategy = "toeplitz"
    scope = 27
    ricker_width = 3
    damp = 0.1
    model = InhibitionNetwork(logdir=f"{strategy}/scope_{scope}/width_{ricker_width}/damp_{damp}",
                              scope=[scope],
                              width=ricker_width,
                              damp=0.1,
                              inhibition_depth=1,
                              inhibition_strategy=strategy,
                              )

    model.load_state_dict(torch.load(
        f"../saved_models/{strategy}/scope_{scope}/width_{ricker_width}/damp_{damp}/InhibitionNetwork_{strategy}_155.model"))
    base = network = Baseline(logdir="test")
    base.load_state_dict(torch.load("../saved_models/test/Baseline_155.model"))
    nets = [model, base]
    for net in nets:
        differences = []
        filters = net.features[0].weight.data.numpy()
        for i in range(len(filters) - 1):
            diff = mse(filters[i + 1], filters[i])
            differences.append(diff)
        print(sum(differences) / len(differences))
        from util.filter_ordering import two_opt

        sorted_filters: List[Tensor] = two_opt(filters)
        show_ordering_difference(filters, sorted_filters)
        # plot_unsorted_and_sorted_filters(filters, sorted_filters)
