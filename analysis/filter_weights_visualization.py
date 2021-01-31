"""helper file for visualizations of filter ordering"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from utilities.image import rgb2gray
from analysis.util import load_model_by_id


def visualize_filters(filters: List[Tensor], grayscale=False, title=None):
    """
    visualizes the given filters
    :param filters:         the filters
    """
    rows, cols = get_dim_for_plot(len(filters))
    fig, axs = plt.subplots(rows, cols)
    f = 0
    for row in range(rows):
        for col in range(cols):
            img = np.swapaxes(filters[f], 0, 2)
            img = np.ndarray.astype(np.interp(img, (img.min(), img.max()), (0, 1)), dtype=float)
            if grayscale and img.shape[-1] == 3:
                img = rgb2gray(img)
            axs[row, col].imshow(img)
            axs[row, col].axis("off")

            f += 1

    if title is not None:
        fig.suptitle(title)


def plot_unsorted_and_sorted_filters(filters: List[Tensor], sorted_filters: List[Tensor]):
    """
    visualizes the given filters

    :param filters:             the filters
    :param sorted_filters:      the sorted filters
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
    returns a list of tuples comparing the indices of filters from the previous ordering with the indices after sorting

    :param filters:             the filters
    :param sorted_filters:      the sorted filters

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
    returns the "best" dimension for plotting filters, e.g. n = 6 returns 2,3 and 16 returns 4,4
    adapted from Daniel Lee's solution (https://stackoverflow.com/questions/39248245/factor-an-integer-to-something-as-close-to-a-square-as-possible/39248503#39248503)

    :param n:       the number of filters

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
        load_model_by_id("1612027536819388"),  # capsnet semlc
        load_model_by_id("1611789986722637"),  # capsnet none

        # ALEXNET
        # load_model_by_id("1611720276269320"),  # alexnet semlc
        # load_model_by_id("1611709223422344"),  # alexnet none
    ]

    show_every_nth = 1

    for model in models:
        weights = model.get_conv_one().weight.data.numpy()
        visualize_filters([f for i, f in enumerate(weights) if i % show_every_nth == 0], grayscale=True, title=model.lateral_type)
        # visualize_filters([f for i, f in enumerate(weights) if i % show_every_nth == 0], grayscale=True, title=model.lateral_type)

    plt.show()
