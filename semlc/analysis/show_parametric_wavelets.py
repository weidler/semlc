import torch

from analysis.util import get_group_model_ids, load_model_by_id
from core.weight_initialization import difference_of_gaussians
from run import generate_group_handle

from matplotlib import pyplot as plt

network = "shallow"
dataset = "cifar10"

ids = get_group_model_ids(generate_group_handle(network, dataset, "parametric-semlc"))

epsp_widths = []
ipsp_widths = []
damps = []

fig, axs = plt.subplots(1, 3)
fig.set_size_inches(24, 6)

mexican_hats = []
inverted_mexican_hats = []
for id in ids:
    model = load_model_by_id(id).lateral_layer
    wavelet = difference_of_gaussians(model.in_channels - 1,
                                      (torch.tensor(model.width_epsps), torch.tensor(model.width_ipsps)),
                                      torch.tensor(model.ratio),
                                      torch.tensor(model.damp)).cpu().detach().numpy()

    epsp_widths.append(model.widths[0])
    ipsp_widths.append(model.widths[1])
    damps.append(model.damp.cpu().detach().numpy().item())

    if not wavelet.max() > 0.1 and not wavelet.min() < -0.1:
        if model.damp > 0:
            mexican_hats.append(wavelet)
            axs[1].plot(wavelet)
        else:
            inverted_mexican_hats.append(wavelet)
            axs[2].plot(wavelet)

        axs[0].plot(wavelet)

axs[0].set_title(f"All Connectivity Profiles")
axs[1].set_title(f"Mexican Hats ({len(mexican_hats)})")
axs[2].set_title(f"Inverted Mexican Hats ({len(inverted_mexican_hats)})")

plt.show()



# wavelet = ricker_wavelet(model.in_channels - 1,
#                          torch.tensor(model.ricker_width).float(),
#                          torch.tensor(model.ricker_damp).float()).cpu().detach().numpy()
# plt.plot(wavelet, color="black")
#
# plt.show()

# plt.hist(widths, 10, density=True)
# plt.show()
#
# plt.hist(damps, 10, density=True)
# plt.show()