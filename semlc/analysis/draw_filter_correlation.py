import numpy

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.distance import cosine

from analysis.util import get_group_model_ids, load_model_by_id
from run import generate_group_handle
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times"
matplotlib.rcParams["font.weight"] = 'normal'
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["axes.labelsize"] = 9
matplotlib.rcParams["axes.titlesize"] = 9
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


def corr_measure(x, y):
    return 1 - cosine(x, y)


network = "alexnet"
dataset = "cifar10"
strat = "semlc"

ids = get_group_model_ids(generate_group_handle(network, dataset, strat))

model = load_model_by_id(ids[10])
filters = model.get_conv_one().weight.data.detach().numpy()
n_filters = filters.shape[0]

tuning_curves = []
for i in range(n_filters):
    focus_filter = filters[i, ...]

    correlation_vector = []
    for f in filters:
        correlation_vector.append(corr_measure(focus_filter.flatten(), f.flatten()))
    correlation_vector = numpy.array(correlation_vector)

    tuning_curves.append(numpy.roll(correlation_vector, (n_filters // 2) - i))

x = numpy.arange(n_filters) - (n_filters // 2)
y = numpy.mean(numpy.array(tuning_curves), axis=0)

# plot
fig: Figure
axs: Axes
fig, axs = plt.subplots()

axs.set_xlim(x.min(), x.max())
axs.set_xlabel("Filter")
axs.set_ylabel("Cosine Similarity")
axs.bar(x.tolist(), y)

fig.set_size_inches(3.25, 2.5)
fig.tight_layout()

plt.savefig(f"{network}-{strat}-tuning-curve.pdf", format="pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0.01)
# plt.show()
