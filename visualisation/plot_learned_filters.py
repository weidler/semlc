"""plots 10 samples of the learned filter for he specified strategy
as well as the mean adaption vs the initialization"""

import numpy
import torch
import matplotlib as mp
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt

from util import ricker
from visualisation.helper import get_one_model, get_net

mp.rcParams['ps.useafm'] = True
mp.rcParams['pdf.use14corefonts'] = True

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

fig: Figure = plt.figure(constrained_layout=False)
fig.set_size_inches(12, 7)
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 2])

models = []
strategy = "converged_full_best"
num_nets = 60
layer = 1
num_plot_samples = 10

for i in range(0, num_nets):
    model = get_one_model(strategy, index=i)
    models.append(model)

net = get_net(strategy)
parameters = list(net.named_parameters())
filter_before = None
damp_before = None
width_before = None
for p in parameters:
    if strategy == "parametric":
        if p[0] == f"features.inhib_{layer}.damp":
            damp_before = p[1].data.cpu()
        if p[0] == f"features.inhib_{layer}.width":
            width_before = p[1].data.cpu()
        if damp_before is not None and width_before is not None:
            filter_before = ricker.ricker(scope=45, width=width_before,
                                          damp=damp_before, self_connect=False).cpu().view(-1).numpy()
            break
    else:
        if p[0] == f"features.inhib_{layer}.inhibition_filter":
            filter_before = p[1].data.cpu().view(-1).numpy()
            break

filters_after = []
for i in range(1, num_plot_samples + 1):
    parameters = list(models[i - 1].named_parameters())
    filter_after = None
    damp_after = None
    width_after = None
    for p in parameters:
        if strategy == "parametric":
            if p[0] == f"features.inhib_{layer}.damp":
                damp_after = p[1].data.cpu()
            if p[0] == f"features.inhib_{layer}.width":
                width_after = p[1].data.cpu()
            if damp_after is not None and width_after is not None:
                filter_after = ricker.ricker(scope=45, width=width_after,
                                             damp=damp_after, self_connect=False).cpu().view(-1).numpy()
        else:
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
        ax.plot(filter_after, color="firebrick")

mean_filter_after = numpy.mean(numpy.asarray(filters_after), axis=0)

gs2 = fig.add_gridspec(3, 5, wspace=0, hspace=0.5, height_ratios=[1, 1, 2])

ax_left: Axes = plt.subplot(gs2[2, :2])
ax_left.set_title("Initialization")
ax_left.set_xlabel("Filter Dimension")
ax_left.set_ylabel("Weight")
ax_left.set_yticks([])
ax_left.set_xticks([])

ax_right: Axes = plt.subplot(gs2[2, 3:])
ax_right.set_title(f"Mean Adaption over all {num_nets} Trainings")
ax_right.set_xlabel("Filter Dimension")
ax_right.set_ylabel("Weight")
ax_right.set_yticks([])
ax_right.set_xticks([])

fig.suptitle("\n\nAdaptions of 10 Independent Trainings")

ax_left.plot(filter_before, color="royalblue")
ax_right.plot(mean_filter_after, color="maroon")

plt.savefig(f"./documentation/figures/adapted_filters_{strategy}_layer_{layer}.pdf", format="pdf", bbox_inches="tight")
plt.show()
