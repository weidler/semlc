import matplotlib
import matplotlib.pyplot as plt
import numpy
import torch

from core import ricker_wavelet

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = "Times"
matplotlib.rcParams["font.weight"] = 'normal'
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["legend.fontsize"] = 7
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

size = 63
fig, axs = plt.subplots()

x = numpy.array(list(range(size))) - (size // 2)
for width in [1., 3., 7., 12.]:
    y_r = ricker_wavelet(size, torch.tensor(width), torch.tensor(0.2)).view(-1).numpy()
    # y_dog = dog_mexican_hat(size,
    #                         (torch.tensor(width), torch.tensor(width * 3)),
    #                         (torch.tensor(1), torch.tensor(2))).view(-1).numpy()
    axs.plot(x, y_r, label=f"width={int(width)}", lw=1)
    # plt.plot(x, y_dog, label=f"DoG: \u03C3={width}")

axs.set_xlabel("Relative Filter Position")
axs.set_ylabel("Weight")
axs.legend()

fig.set_size_inches(3.25, 2)
fig.tight_layout()

fig.savefig("../documentation/figures/wavelet.pdf", format="pdf", dpi=plt.gcf().dpi, bbox_inches="tight", pad_inches=0.01)