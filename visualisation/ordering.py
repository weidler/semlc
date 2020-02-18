"""visually compares the ordering imposed by the network with the suggested ordering by the two-opt"""

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

# 30 for parametric else 60
num_nets = 30
num_layer = 0

# strategies = ['parametric' for _ in range(num_nets)]
strategies = ['baseline', 'cmap', 'converged', 'converged_freeze', 'parametric']
names = ["None", "LRN", "CLC Adaptive", "CLC Frozen", "CLC Parametric"]

models = []

indices = {
    'parametric': 12,
    'converged': 28,
    'converged_freeze': 2,
    'baseline': 0,
    'cmap': 0
}

for i, strategy in enumerate(strategies):
    model = get_one_model(strategy, index=indices[strategy])
    # model = get_one_model(strategy, index=i)
    models.append(model)

print(models[0].features[num_layer])

bc, ssi_c, conv_c = plt.get_cmap("Set1").colors[:3]

# use for comparison of 5 strategies
colors = [bc, bc, conv_c, conv_c, conv_c]

# use for plotting all models of single strategy
# colors = [conv_c for _ in range(num_nets)]

# use for plotting all models of single strategy
# r, c = get_dim_for_plot(len(strategies))

# use for comparison of 5 strategies
r, c = 1, 5

print(r, c)
gs = gridspec.GridSpec(r, c)
fig, axs = plt.subplots(r, c)

# use for comparison of 5 strategies
fig.set_size_inches(8, 2)
# fig.set_size_inches(14, 14)

# use for plotting all models of single strategy
# rows = [i for i in range(r) for _ in range(c)]
# cols = [i for _ in range(r) for i in range(c)]

# use for comparison of 5 strategies
rows = [0, 0, 0, 0, 0]
cols = [0, 1, 2, 3, 4]

print(rows)
print(cols)
counter = 0


for net in tqdm(models, disable=False):
    filters = net.features[num_layer].weight.data.numpy()

    plot_row = rows[counter]
    plot_col = cols[counter]
    ax: Axes = plt.subplot(gs[plot_row, plot_col])
    # if plot_col != 0:
    ax.set_yticks([])
    # if plot_row != r - 1:
    ax.set_xticks([])
    create_plot(net, ax, cmap=colors[counter], num_layer=num_layer, point_size=2)
    ax.set_title(names[counter])

    counter += 1

# use for plotting all models of single strategy
# fig.savefig(f'./documentation/figures/ordering_{strategies[0]}.pdf', format="pdf", bbox_inches='tight')

# use for comparison of 5 strategies
fig.savefig(f'./documentation/figures/ordering.pdf', format="pdf", bbox_inches='tight')
plt.show()
