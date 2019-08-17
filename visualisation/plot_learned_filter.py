from typing import List, Any, Tuple

import numpy
import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from model.network.alexnet_paper import ConvergedInhibitionNetwork, SingleShotInhibitionNetwork

import matplotlib.pyplot as plt

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

fig: Figure = plt.figure(constrained_layout=False)
fig.set_size_inches(12, 7)
gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 2])

# net = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1)
net = SingleShotInhibitionNetwork([63], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1)
parameters = list(net.named_parameters())
filter_before = None
for p in parameters:
    if p[0] == "features.inhib_1.convolver.weight":
        filter_before = p[1].data.cpu().view(-1).numpy()
        break

filters_after = []
for i in range(1, 11):
    filename = f"../final_results/ss/ss_{i}/{net.__class__.__name__}_best.model"
    net.load_state_dict(torch.load(filename))

    parameters = list(net.named_parameters())
    filter_after = None
    for p in parameters:
        if p[0] == "features.inhib_1.convolver.weight":
            filter_after = p[1].data.cpu().view(-1).numpy()
            break

    filters_after.append(filter_after)

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

ax_right: Axes = plt.subplot(gs2[2, 3:])
ax_right.set_title("Mean Adaption")
ax_right.set_xlabel("Filter Dimension")
ax_right.set_ylabel("Weight")

fig.suptitle("\n\nAdaptions of 10 Independent Trainings")

ax_left.plot(filter_before, color="royalblue")
ax_right.plot(mean_filter_after, color="maroon")

plt.savefig("../documentation/figures/adapted_filters_ss.pdf", format="pdf", bbox_inches="tight")
plt.show()