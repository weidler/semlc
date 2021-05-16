"""visually compares the ordering imposed by the networks with the suggested ordering by the two-opt"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from tqdm import tqdm

from analysis.util import load_model_by_id, get_group_model_ids
from analysis import create_plot

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

group = None

network = "simple"
dataset = "cifar10"

num_layer = 0


models = []

groups = [
    f'{network}-{dataset}',
    f'{network}-{dataset}-lrn',
    f'{network}-{dataset}-semlc',
    f'{network}-{dataset}-parametric-semlc',
    f'{network}-{dataset}-adaptive-semlc',
    f'{network}-{dataset}-gaussian-semlc'
]
colors = plt.get_cmap("tab10").colors[:len(groups)]
r, c = 1, len(groups)

for i, group in enumerate(groups):
    try:
        model = load_model_by_id(get_group_model_ids(group)[22])
        models.append(model)
    except IndexError:
        pass

rows = [0, 0, 1, 1, 2, 2]
cols = [0, 1, 0, 1, 0, 1]

gs = gridspec.GridSpec(max(rows) + 1, max(cols) + 1)
fig, axs = plt.subplots(r, c)

counter = 0
for net in tqdm(models, disable=False):
    filters = net.get_conv_one().weight.data.numpy()

    plot_row = rows[counter]
    plot_col = cols[counter]
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    ax.set_yticks([])
    ax.set_xticks([])
    create_plot(net, ax, cmap=colors[counter], num_layer=num_layer, point_size=1)
    ax.set_title(net.lateral_layer.name if hasattr(net, "lateral_layer") else "Baseline")

    counter += 1

fig.set_size_inches(3.25, 3.5)
fig.tight_layout()

fig.savefig(f'ordering-{network}-{dataset}.pdf', format="pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)

plt.show()
