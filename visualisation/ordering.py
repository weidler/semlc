"""visually compares the ordering imposed by the networks with the suggested ordering by the two-opt"""

import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.axes import Axes
from tqdm import tqdm

from visualisation.filter_weights_visualization import get_dim_for_plot
from visualisation.helper import get_one_model
from visualisation.plot_ordering import create_plot

mp.rcParams['ps.useafm'] = True
mp.rcParams['pdf.use14corefonts'] = True
mp.rcParams['text.usetex'] = True

# set strategy to plot all its models' ordering
# set strategy to None to create comparison plot in paper
# strategy = 'cmap'
strategy = None

num_layer = 0
bc, ssi_c, conv_c = plt.get_cmap("Set1").colors[:3]

models = []

if strategy is not None:
    num_nets = 30 if strategy == 'parametric' else 60
    strategies = [strategy for _ in range(num_nets)]
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
    colors = [color_map[strategy] for _ in range(num_nets)]
    r, c = get_dim_for_plot(len(strategies))

else:
    strategies = ['baseline', 'cmap', 'converged', 'converged_freeze', 'parametric']
    names = ["None", "LRN", "CLC Adaptive", "CLC Frozen", "CLC Parametric"]

    colors = [bc, bc, conv_c, conv_c, conv_c]
    r, c = 1, 5

for i, strat in enumerate(strategies):
    if strategy is not None:
        model = get_one_model(strat, index=i)
    else:
        indices = {
            'parametric': 12,
            'converged': 28,
            'converged_freeze': 2,
            'baseline': 0,
            'cmap': 0
        }
        model = get_one_model(strat, index=indices[strat])

    models.append(model)


gs = gridspec.GridSpec(r, c)
fig, axs = plt.subplots(r, c)

if strategy is not None:
    fig.set_size_inches(14, 14)
    rows = [i for i in range(r) for _ in range(c)]
    cols = [i for _ in range(r) for i in range(c)]

else:
    fig.set_size_inches(8, 2)
    rows = [0, 0, 0, 0, 0]
    cols = [0, 1, 2, 3, 4]

counter = 0
for net in tqdm(models, disable=False):
    filters = net.features[num_layer].weight.data.numpy()

    plot_row = rows[counter]
    plot_col = cols[counter]
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    ax.set_yticks([])
    ax.set_xticks([])
    create_plot(net, ax, cmap=colors[counter], num_layer=num_layer, point_size=2)
    if names is not None:
        ax.set_title(names[counter])

    counter += 1

if strategy is not None:
    # use for plotting all models of single strategy
    fig.savefig(f'./documentation/figures/ordering_{strategy}.pdf', format="pdf", bbox_inches='tight')

else:
    # use for comparison of 5 strategies
    fig.savefig(f'./documentation/figures/ordering.pdf', format="pdf", bbox_inches='tight')

plt.show()
