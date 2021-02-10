"""helper file for visualizations of filter ordering"""

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import make_grid

from analysis.util import load_model_by_id
from utilities import rgb2gray


def visualize_filters(filters_per_group: Dict[str, Tensor], grayscale=False):
    """ Visualizes the given weights in a grid
    :param filters_per_group:         the weights
    """

    fig, axs = plt.subplots(1, len(filters_per_group))
    i = 0
    for group, filters in filters_per_group.items():
        rows, cols = get_dim_for_plot(len(filters))
        grid = make_grid(filters, padding=2, nrow=rows, pad_value=filters.max()).numpy().transpose(1, 2, 0)
        grid = rgb2gray(grid) if grayscale else grid
        axs[i].imshow(grid)
        axs[i].axis("off")
        axs[i].set_title(group)
        i += 1


def plot_unsorted_and_sorted_filters(filters: List[Tensor], sorted_filters: List[Tensor]):
    """
    visualizes the given filters_per_group

    :param filters:             the filters_per_group
    :param sorted_filters:      the sorted filters_per_group
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


def get_ordering_difference(filters: List[Tensor], sorted_filters: List[Tensor]):
    """
    returns a list of tuples comparing the indices of filters_per_group from the previous ordering with the indices after sorting

    :param filters:             the filters_per_group
    :param sorted_filters:      the sorted filters_per_group

    :return                     a list of tuples containing the difference in indices
    """
    diff = []
    for i in range(len(filters)):
        for j in range(len(filters)):
            if np.all(filters[i] == sorted_filters[j]):
                diff.append([i + 1, j + 1])
    return diff


def get_dim_for_plot(n):
    """
    returns the "best" dimension for plotting filters_per_group, e.g. n = 6 returns 2,3 and 16 returns 4,4
    adapted from Daniel Lee's solution (https://stackoverflow.com/questions/39248245/factor-an-integer-to-something-as-close-to-a-square-as-possible/39248503#39248503)

    :param n:       the number of filters_per_group

    :return:        a tuple of rows and cols for the plot
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


if __name__ == '__main__':
    models = [
        # CAPSNET
        load_model_by_id("1612174588327432"),  # capsnet none
        load_model_by_id("1612119956839388"),  # capsnet semlc

        # ALEXNET
        # load_model_by_id("1611720276269320"),  # alexnet semlc
        # load_model_by_id("1611709223422344"),  # alexnet none
    ]

    show_every_nth = 1

    weights: Dict[str, torch.Tensor] = {}
    for model in models:
        fs = model.get_conv_one().weight.data
        # for fid in range(fs.shape[0]):
        #     fs[fid, ...] = (fs[fid, ...] - fs[fid, ...].max()) / (fs[fid, ...].max() - fs[fid, ...].min())

        for fid in range(fs.shape[0]):
            fs[fid, ...] = (fs[fid, ...] - fs[fid, ...].mean()) / fs[fid, ...].std()
        weights[str(model.lateral_type)] = fs

    visualize_filters(weights, grayscale=True)
    plt.show()
