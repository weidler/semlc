import matplotlib.pyplot as plt
import numpy
import torch

from core.weight_initialization import ricker_wavelet, dog_mexican_hat

size = 63

x = numpy.array(list(range(size))) - (size // 2)
for width in reversed([3.]):
    y_r = ricker_wavelet(size, torch.tensor(width), torch.tensor(0.2)).view(-1).numpy()
    y_dog = dog_mexican_hat(size,
                            (torch.tensor(width), torch.tensor(width * 3)),
                            (torch.tensor(1), torch.tensor(2))).view(-1).numpy()
    plt.plot(x, y_r, label=f"Ricker: \u03C3={width}")
    plt.plot(x, y_dog, label=f"DoG: \u03C3={width}")

plt.xlabel("Relative Filter Position")
plt.ylabel("Weight")

plt.legend()
plt.savefig("../documentation/figures/wavelet.pdf", format="pdf", bbox_inches="tight")
plt.show()
