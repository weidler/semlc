from typing import List
import matplotlib.pyplot as plt

import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from torch import Tensor
from tqdm import tqdm

from visualisation.plot_ordering import get_orderings, create_plot
from model.network.alexnet_cifar import ConvergedInhibitionNetwork, Baseline, BaselineCMap
from visualisation.filter_weights_visualization import get_ordering_difference
from util.filter_ordering import two_opt

gs = gridspec.GridSpec(1, 4)
fig, axs = plt.subplots(1, 4)
fig.set_size_inches(14, 4)
ricker_width = 8
damp = 0.1

base = Baseline()
cmap = BaselineCMap()
converged = ConvergedInhibitionNetwork(scopes=[27], width=ricker_width, damp=0.1, freeze=False)
converged_f = ConvergedInhibitionNetwork(scopes=[27], width=ricker_width, damp=0.1)

base.load_state_dict(torch.load(
    f"../final_results/baseline/baseline_1/Baseline_best.model",
    map_location=lambda storage, loc: storage))
cmap.load_state_dict(torch.load(
    f"../final_results/cmap_models/cmap_1/BaselineCMap_final.model",
    map_location=lambda storage, loc: storage))
converged_f.load_state_dict(torch.load(
    f"../final_results/converged_freeze/converged_freeze_1/ConvergedInhibitionNetwork_freeze_best.model",
    map_location=lambda storage, loc: storage))
converged.load_state_dict(torch.load(
    f"../final_results/converged/converged_2/ConvergedInhibitionNetwork_best.model",
    map_location=lambda storage, loc: storage))

names = ["Baseline", "Baseline + LRN", "Converged Adaptive", "Converged Frozen"]
rows = [0, 0, 0, 0]
cols = [0, 1, 2, 3]
counter = 0
for net in tqdm([base, cmap, converged, converged_f]):
    filters = net.features[0].weight.data.numpy()
    sorted_filters: List[Tensor] = two_opt(filters)
    diff = get_ordering_difference(filters, sorted_filters)
    original_ordering, inhibited_ordering = get_orderings(net)

    plot_row = rows[counter]
    plot_col = cols[counter]
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    create_plot(net, ax)
    ax.set_title(names[counter])
    counter += 1
# fig.savefig('../documentation/figures/four_filter_ordering.pdf', format="pdf", bbox_inches='tight')
plt.show()


