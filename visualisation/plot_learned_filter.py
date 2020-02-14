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
fig.set_size_inches(12, 7)
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 2])

models = []
strategy = "parametric"
num_nets = 1
layer = 1
num_plot_samples = 10

for i in range(0, num_nets):
    # get first model found
    model = get_one_model(strategy, index=i)
    models.append(model)

print(models[0].features)
# net = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1)
net = get_net(strategy)
parameters = list(net.named_parameters())
filter_before = None
for p in parameters:
    if p[0] == f"features.inhib_{layer}.inhibition_filter":
        filter_before = p[1].data.cpu().view(-1).numpy()
        break

filters_after = []
for i in range(1, num_plot_samples + 1):
    parameters = list(models[i - 1].named_parameters())
    # print(parameters)
    filter_after = None
    for p in parameters:
        if p[0] == f"features.inhib_{layer}.inhibition_filter":
            filter_after = p[1].data.cpu().view(-1).numpy()
            break

    filters_after.append(filter_after)

    if i < num_plot_samples + 1:
        plot_row = 0 if i < 6 else 1
        plot_col = (i % 5) - 1
        ax: Axes = plt.subplot(gs[plot_row, plot_col])
        ax.set_yticks([])
        ax.set_xticks([])
        print(filter_after)
        ax.plot(filter_after, color="firebrick")

mean_filter_after = numpy.mean(numpy.asarray(filters_after), axis=0)

gs2 = fig.add_gridspec(3, 5, wspace=0, hspace=0.5, height_ratios=[1, 1, 2])

ax_left: Axes = plt.subplot(gs2[2, :2])
ax_left.set_title("Initialization")
ax_left.set_xlabel("Filter Dimension")
ax_left.set_ylabel("Weight")

ax_right: Axes = plt.subplot(gs2[2, 3:])
ax_right.set_title(f"Mean Adaption over all {num_nets} Trainings")
ax_right.set_xlabel("Filter Dimension")
ax_right.set_ylabel("Weight")

fig.suptitle("\n\nAdaptions of 10 Independent Trainings")

ax_left.plot(filter_before, color="royalblue")
ax_right.plot(mean_filter_after, color="maroon")

plt.savefig(f"../documentation/figures/adapted_filters_{strategy}_layer_{layer}.pdf", format="pdf", bbox_inches="tight")
plt.show()
