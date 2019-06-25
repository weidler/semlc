from typing import List, Any, Tuple

import numpy
import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from model.network.alexnet_paper import ConvergedInhibitionNetwork

import matplotlib.pyplot as plt

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

fig: Figure = plt.gcf()
fig.set_size_inches(12, 7)
gs = gridspec.GridSpec(3, 5)

net = ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, inhibition_start=1, inhibition_end=1)
parameters = list(net.named_parameters())
filter_before = None
for p in parameters:
    if p[0] == "features.inhib_1.inhibition_filter":
        filter_before = p[1].data.cpu().view(-1).numpy()
        break

filters_after = []
for i in range(1, 11):
    filename = f"../final_results/converged/converged_{i}/{net.__class__.__name__}_best.model"
    net.load_state_dict(torch.load(filename))

    parameters = list(net.named_parameters())
    filter_after = None
    for p in parameters:
        if p[0] == "features.inhib_1.inhibition_filter":
            filter_after = p[1].data.cpu().view(-1).numpy()
            break

    filters_after.append(filter_after)

    plot_row = 0 if i < 6 else 1
    plot_col = (i % 5) - 1
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.plot(filter_after)

mean_filter_after = numpy.mean(numpy.asarray(filters_after), axis=0)

ax_left: Axes = plt.subplot(gs[2:, :2])
ax_left.set_title("Initialization")
ax_right: Axes = plt.subplot(gs[2:, 3:])
ax_right.set_title("Adaption")

ax_left.plot(filter_before, color="blue")
ax_right.plot(mean_filter_after, color="red")
plt.show()