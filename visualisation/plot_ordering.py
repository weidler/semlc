"""helper file for visualizations of filter ordering"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from util.filter_ordering import two_opt, mse
from visualisation.filter_weights_visualization import get_ordering_difference


def create_plot(net, part, cmap, plot_sequence=False, num_layer=0, point_size=4):
    """
    creates one plot containing the original filter sequence against the suggested filter sequence by the 2-opt algorithm

    :param net:                 the networks
    :param part:                the part of the plot
    :param cmap:                the color map
    :param plot_sequence:       whether to emphasize ordered sub sequences or not
    :param num_layer:           the index of the layers in the networks features
    :param point_size:          the size of the plotted points

    """
    original_ordering, inhibited_ordering = get_orderings(net, num_layer)
    part.plot(original_ordering, inhibited_ordering, "--", linewidth=0.5, alpha=0.5, color=cmap)

    if plot_sequence:
        min_sequence_length = 3

        current_sequence = [inhibited_ordering[0]]
        current_sequence_xs = [1]
        current_direction = -1
        for i, position in enumerate(inhibited_ordering[1:], 2):
            direction = position - current_sequence[-1]
            # check if same direction
            if ((direction == current_direction) and (direction == 0)) or (direction * current_direction > 0):
                current_sequence.append(position)
                current_sequence_xs.append(i)
            else:
                if len(current_sequence) >= min_sequence_length:
                    part.plot(current_sequence_xs, current_sequence,
                              color="red" if current_direction == -1 else "green")
                current_sequence = [current_sequence[-1], position]
                current_sequence_xs = [current_sequence_xs[-1], i]
                current_direction = direction / abs(direction)

    part.scatter(original_ordering, inhibited_ordering, s=point_size, color=cmap)


def plot_ordering(net, plot_sequence=False, num_layer=0, save=True, point_size=4):
    fig, _ = plt.subplots()
    create_plot(net, plt, plot_sequence=plot_sequence, num_layer=num_layer, point_size=point_size)
    if save:
        fig.savefig('./documentation/figures/_ordering.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def get_orderings(net, num_layer=0):
    """
    returns the indices of the original ordering along with the indices of the suggested ordering of the two-opt.
    Example: Filter 1 is found to be correctly placed in the sequence while filter 3 is suggested
    to be a neighbour of Filter 1 (i.e. to be at index 2) and vice versa:
    The function therefore returns ([1, 2, 3, ...], [1, 3, 2, ...]); filter 2 is now at index 3 and vice versa.

    :param net:             the networks
    :param num_layer:       the index of the layers in the networks features

    :return:                a tuple with the original and
    """
    filters = net.features[num_layer].weight.data.numpy()
    sorted_filters: List[Tensor] = two_opt(filters)
    diff = get_ordering_difference(filters, sorted_filters)
    original_ordering = [elem[0] for elem in diff]
    inhibited_ordering = [elem[1] for elem in diff]
    return original_ordering, inhibited_ordering


def mse_difference(filters, scaler=None):
    """
    calculates the mse differences between filters

    :param filters:         a tensor of filters (C X H X W) where C is the number of filters and H and W are spatial dimensions.
    :param scaler:          an optional sklearn scaler to transform the differences

    :return:                the mse difference between filters
    """
    differences = []
    for i in range(-1, len(filters) - 1):
        diff = mse(filters[i + 1], filters[i])
        differences.append(diff)
    if scaler is not None:
        differences = scaler.transform(np.array(differences).reshape(-1, 1))[:, 0].tolist()
    return sum(differences) / len(differences)
