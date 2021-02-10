from analysis.util import get_group_model_ids, load_model_by_id
from core import ricker_wavelet
from run import generate_group_handle

from matplotlib import pyplot as plt

network = "shallow"
dataset = "cifar10"

ids = get_group_model_ids(generate_group_handle(network, dataset, "parametric-semlc"))

widths = []
damps = []
for id in ids:
    model = load_model_by_id(id).lateral_layer
    wavelet = ricker_wavelet(model.in_channels - 1, model.width, model.damp).cpu().detach().numpy()

    widths.append(model.width.cpu().detach().numpy().item())
    damps.append(model.damp.cpu().detach().numpy().item())
    # plt.plot(wavelet)

# wavelet = ricker_wavelet(model.in_channels - 1,
#                          torch.tensor(model.ricker_width).float(),
#                          torch.tensor(model.ricker_damp).float()).cpu().detach().numpy()
# plt.plot(wavelet, color="black")

# plt.show()

plt.hist(widths, 10, density=True)
plt.show()

plt.hist(damps, 10, density=True)
plt.show()