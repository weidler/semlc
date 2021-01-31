import matplotlib.pyplot as plt
import numpy
import torch

from core.weight_initialization import ricker_wavelet

size = 255

x = numpy.array(list(range(size))) - (size // 2)
for width in reversed([3., 12., 32.]):
    y = ricker_wavelet(size, torch.tensor(width), torch.tensor(0.2)).view(-1).numpy()
    plt.bar(x, y, label=f"\u03C3={width}")

plt.xlabel("Relative Filter Position")
plt.ylabel("Weight")

plt.legend()
plt.savefig("../documentation/figures/wavelet.pdf", format="pdf", bbox_inches="tight")
plt.show()
