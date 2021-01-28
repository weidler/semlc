"""visually compares the ordering imposed by the networks with the suggested ordering by the two-opt"""
import random

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from tqdm import tqdm

from visualisation.filter_weights_visualization import get_dim_for_plot
from visualisation.util import load_model_by_id, get_group_model_ids
from visualisation.plot_ordering import create_plot

# mp.rcParams['ps.useafm'] = True
# mp.rcParams['pdf.use14corefonts'] = True
# mp.rcParams['text.usetex'] = True

# set group to plot all its models' ordering
# set group to None to create comparison plot in paper
# group = 'cmap'
group = None

network = "capsnet"
dataset = "mnist"

num_layer = 0
bc, ssi_c, conv_c = plt.get_cmap("Set1").colors[:3]

models = []

if group is not None:
    num_nets = 30 if group == 'parametric' else 60
    groups = [group for _ in range(num_nets)]
    names = None

    color_map = {
        'ss': ssi_c,
        'ss_freeze': ssi_c,
        'converged': conv_c,
        'converged_full_best': conv_c,
        'converged_freeze': conv_c,
        'baseline': bc,
        'cmap': bc
    }
    colors = [color_map[group] for _ in range(num_nets)]
    r, c = get_dim_for_plot(len(groups))

else:
    groups = [f'{network}-{dataset}', f'{network}-{dataset}-lrn', f'{network}-{dataset}-semlc', f'{network}-{dataset}-parametric-semlc',
              f'{network}-{dataset}-adaptive-semlc']
    colors = plt.get_cmap("Set1").colors[:len(groups)]
    r, c = 1, len(groups)

for i, group in enumerate(groups):
    try:
        model = load_model_by_id(random.choice(get_group_model_ids(group)))
        models.append(model)
    except IndexError:
        pass


gs = gridspec.GridSpec(r, c)
fig, axs = plt.subplots(r, c)

if group is not None:
    fig.set_size_inches(14, 14)
    rows = [i for i in range(r) for _ in range(c)]
    cols = [i for _ in range(r) for i in range(c)]

else:
    fig.set_size_inches(8, 2)
    rows = [0, 0, 0, 0, 0]
    cols = [0, 1, 2, 3, 4]

counter = 0
for net in tqdm(models, disable=False):
    filters = net.get_conv_one().weight.data.numpy()

    plot_row = rows[counter]
    plot_col = cols[counter]
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    ax.set_yticks([])
    ax.set_xticks([])
    create_plot(net, ax, cmap=colors[counter], num_layer=num_layer, point_size=2)
    ax.set_title(net.lateral_layer.name if hasattr(net, "lateral_layer") else "No LC")

    counter += 1

if group is not None:
    # use for plotting all models of single group
    fig.savefig(f'../documentation/figures/ordering_{group}.pdf', format="pdf", bbox_inches='tight')

else:
    # use for comparison of 5 groups
    fig.savefig(f'../documentation/figures/ordering-{network}-{dataset}.pdf', format="pdf", bbox_inches='tight')

plt.show()
