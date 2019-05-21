import matplotlib.pyplot as plt
import numpy as np
from typing import List
from torch import Tensor

from model.network.classification import InhibitionClassificationCNN


def visualize_filters(filters: List[Tensor], rows: int, cols: int):
    """
    visualizes the given filters, rows * cols must be >= number of filters
    :param filters: the filters
    :param rows: number of rows in plot
    :param cols: number of cols in plot
    """
    fig, axs = plt.subplots(rows, cols)
    f = 0
    for row in range(rows):
        for col in range(cols):
            img = np.swapaxes(filters[f], 0, 2)
            img = np.ndarray.astype(np.interp(img, (img.min(), img.max()), (0, 1)), dtype=float)
            axs[row, col].imshow(img)
            f += 1
    plt.show()


def plot_unsorted_and_sorted_filters(filters: List[Tensor], sorted_filters: List[Tensor], rows: int, cols: int):
    """
    visualizes the given filters, rows * cols must be >= number of filters
    :param filters: the filters
    :param sorted_filters: the sorted filters
    :param rows: number of rows in plot
    :param cols: number of cols in plot
    """
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
            if np.allclose(filters[i], sorted_filters[j]):
                print(i + 1, j + 1)


if __name__ == "__main__":
    # how to use
    model = InhibitionClassificationCNN()
    show_ordering_difference(model.get_filters_from_layer(0), model.sort_filters_in_layer(0))
    #visualize_filters(model.sort_filters_in_layer(0), 2, 3)
    plot_unsorted_and_sorted_filters(model.features[0].weight.data.numpy(), model.sort_filters_in_layer(0), 2, 3)

