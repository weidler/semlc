"""helper file for visualizations of filter ordering"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor


def visualize_filters(filters: List[Tensor]):
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
            axs[row, col].imshow(img)
            f += 1
    plt.show()


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
