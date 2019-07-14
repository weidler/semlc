from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from torch import Tensor
from tqdm import tqdm

from model.network.alexnet_paper import SingleShotInhibitionNetwork, BaselineCMap
from util.filter_ordering import two_opt, mse
from visualisation.filter_weights_visualization import get_ordering_difference


def create_plot(net, part, plot_sequence=False, num_layer=0, point_size=4):
    original_ordering, inhibited_ordering = get_orderings(net, num_layer)
    part.plot(original_ordering, inhibited_ordering, "--", linewidth=1)

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

    part.scatter(original_ordering, inhibited_ordering, s=point_size)


def plot_ordering(net, plot_sequence=False, num_layer=0, save=True, point_size=4):
    fig, _ = plt.subplots()
    create_plot(net, plt, plot_sequence=plot_sequence, num_layer=num_layer, point_size=point_size)
    if save:
        fig.savefig('../documentation/figures/ordering.pdf', format="pdf", bbox_inches='tight')
    plt.show()


def get_orderings(net, num_layer=0):
    filters = net.features[num_layer].weight.data.numpy()
    sorted_filters: List[Tensor] = two_opt(filters)
    diff = get_ordering_difference(filters, sorted_filters)
    original_ordering = [elem[0] for elem in diff]
    inhibited_ordering = [elem[1] for elem in diff]
    return original_ordering, inhibited_ordering


def mse_difference(filters):
    differences = []
    for i in range(len(filters) - 1):
        diff = mse(filters[i + 1], filters[i])
        differences.append(diff)

    return sum(differences) / len(differences)


def get_mean_difference(net):
    filters = net.features[0].weight.data.numpy()
    mean = mse_difference(filters)

    np.random.shuffle(filters)
    sorted_filters: List[Tensor] = two_opt(filters)

    two_opt_mean = mse_difference(sorted_filters)

    return mean, two_opt_mean


def plot_all(save=True):
    gs = gridspec.GridSpec(2, 5)
    fig, axs = plt.subplots(2, 5)
    fig.set_size_inches(10, 5)
    strategy = "cmap"
    scope = 27
    ricker_width = 8
    damp = 0.1
    # model = ConvergedInhibitionNetwork(
    # model = SingleShotInhibitionNetwork(
    #    scopes=[scope],
    #    width=ricker_width,
    #    damp=0.1,
    #    freeze=False,
    # )
    net = BaselineCMap()
    for j in tqdm(range(1, 11)):
        if j < 11:
            net.load_state_dict(torch.load(
                # f"../final_results/{strategy}/{strategy}_{j}/ConvergedInhibitionNetwork_best.model",
                # f"../final_results/{strategy}_models/{strategy}_{j}/SingleShotInhibitionNetwork_freeze_best.model",
                f"../final_results/{strategy}_models/{strategy}_{j}/BaselineCMap_final.model",
                map_location=lambda storage, loc: storage))
        else:
            scope = 63
            net = SingleShotInhibitionNetwork(
                scopes=[scope],
                width=ricker_width,
                damp=0.1
            )
            net.load_state_dict(torch.load(
                f"../final_results/ss/ss_{j - 10}/SingleShotInhibitionNetwork_best.model",
                map_location=lambda storage, loc: storage))

        plot_row = 0 if j < 6 else 1 if j < 11 else 0 if j < 16 else 1
        plot_col = (j % 5) - 1
        ax: Axes = plt.subplot(gs[plot_row, plot_col])
        create_plot(net, ax)
    fig.suptitle("Baseline + LRN")
    if save:
        fig.savefig('../documentation/figures/baselinecmap_ordering.pdf', format="pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # plot_all(save=False)

    strategy = "cmap"
    model = BaselineCMap()
    plot_ordering(model, save=False, point_size=20)
    for i in tqdm(range(1, 2)):
        model.load_state_dict(torch.load(f"../final_results/{strategy}_models/{strategy}_{i}/BaselineCMap_final.model",
                                         map_location=lambda storage, loc: storage))
        orig, two_opt = get_mean_difference(model)
        print('model', orig)
        print('2-opt', two_opt)
