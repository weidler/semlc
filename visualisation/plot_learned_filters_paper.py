from typing import List, Any, Tuple

import numpy
import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from model.network.alexnet_cifar import ConvergedInhibitionNetwork, SingleShotInhibitionNetwork

import matplotlib.pyplot as plt

from visualisation.helper import get_one_model, get_net

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

fig: Figure = plt.figure(constrained_layout=False)
fig.set_size_inches(7.3, 4.8)
gs = fig.add_gridspec(2, 2)

models = []
strategy = "converged_full_best"
num_nets = 60
layers = [1, 2, 3, 4]

for i in range(0, num_nets):
    # get first model found
    model = get_one_model(strategy, index=i)
    models.append(model)

# net = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1)
bc, ssi_c, conv_c = plt.get_cmap("Set1").colors[:3]
colors = [conv_c for _ in range(len(layers))]
rows = [0, 0, 1, 1]
cols = [0, 1, 0, 1]

net = get_net(strategy)
orig_parameters = list(net.named_parameters())

# all_filters_after = []


for layer in range(1, len(layers) + 1):
    filters_after = []
    filter_before = None
    for p in orig_parameters:
        if p[0] == f"features.inhib_{layer}.inhibition_filter":
            filter_before = p[1].data.cpu().view(-1).numpy()
            break

    for i in range(1, len(models) + 1):

        parameters = list(models[i - 1].named_parameters())
        # print(parameters)
        filter_after = None
        for p in parameters:
            if p[0] == f"features.inhib_{layer}.inhibition_filter":
                filter_after = p[1].data.cpu().view(-1).numpy()
                filters_after.append(filter_after)
                break

    mean_filter_after = numpy.mean(numpy.asarray(filters_after), axis=0)
    plot_row = rows[layer - 1]
    plot_col = cols[layer - 1]
    print(plot_row, plot_col)
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    ax.set_yticks([])
    ax.set_xticks([])
    # ax.set_title("Wavelet")

    '''
    if plot_col == 0:
        ax.set_ylabel("Weight")
    if plot_row == 1:
        ax.set_xlabel("Filter Dimension")
    '''
    ax.set_ylim(-0.2, 0.1)

    ax.plot(filter_before, label="Initialization")
    ax.plot(mean_filter_after, label=f"Mean Adaption over all {num_nets} Trainings")

    '''
    if i < num_plot_samples + 1:
        plot_row = 0 if i < 6 else 1
        plot_col = (i % 5) - 1
        ax: Axes = plt.subplot(gs[plot_row, plot_col])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.plot(filter_after, color="firebrick")
    '''
    # all_filters_after.append(filters_after)

# mean_filter_after = numpy.mean(numpy.asarray(filters_after), axis=0)
# fig.suptitle("Converged Adaptive Full", y=0.95)
# fig.subplots_adjust(bottom=0.2)



'''
ax_right: Axes = plt.subplot(gs2[2, 3:])
ax_right.set_title(f"Mean Adaption over all {num_nets} Trainings")

ax_right.set_xlabel("Filter Dimension")
ax_right.set_ylabel("Weight")

fig.suptitle("\n\nAdaptions of 10 Independent Trainings")
'''
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)

plt.savefig(f"../documentation/figures/adapted_filters.pdf", format="pdf", bbox_inches="tight")
plt.show()
