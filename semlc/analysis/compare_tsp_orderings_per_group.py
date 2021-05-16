"""visually compares the ordering imposed by the networks with the suggested ordering by the two-opt"""
import random

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from analysis.util import load_model_by_id, get_group_model_ids
from analysis.plot_ordering import create_plot

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times"
matplotlib.rcParams["font.weight"] = 'normal'
matplotlib.rcParams["font.size"] = 7
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 7
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams['axes.unicode_minus'] = False

matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 2
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['xtick.minor.size'] = 2
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.major.size'] = 2
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['ytick.minor.size'] = 2
matplotlib.rcParams['ytick.minor.width'] = 0.5
matplotlib.rcParams['axes.unicode_minus'] = False

network = "simple"
dataset = "cifar10"

n_networks = 30
r, c = 6, 5


for i, group in enumerate([f'{network}-{dataset}', f'{network}-{dataset}-lrn', f'{network}-{dataset}-semlc',
              f'{network}-{dataset}-parametric-semlc', f'{network}-{dataset}-adaptive-semlc',
              f'{network}-{dataset}-gaussian-semlc']):

    color = plt.get_cmap("tab10").colors[i]
    model_ids = random.sample(get_group_model_ids(group), n_networks)

    models = []
    for id in model_ids:
        try:
            models.append(load_model_by_id(id))
        except IndexError:
            print(f"missed model {id}")
            pass

    fig, axs = plt.subplots(r, c)

    for counter, net in tqdm(list(enumerate(models)), disable=False, desc=group):
        filters = net.get_conv_one().weight.data.numpy()

        plot_row = counter // c
        plot_col = counter % c

        ax = axs[plot_row][plot_col]
        ax.set_yticks([])
        ax.set_xticks([])
        create_plot(net, ax, cmap=color, num_layer=0, point_size=1)

    fig.set_size_inches(6.75, 7.75)
    fig.tight_layout()

    fig.savefig(f'ordering-multiple-{group}.pdf', format="pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
