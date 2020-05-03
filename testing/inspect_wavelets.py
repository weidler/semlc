from util.weight_initialization import mexican_hat
import torch
import matplotlib.pyplot as plt
import numpy

wavelet_64_clca = mexican_hat(31, torch.tensor(3.0), torch.tensor(0.1), False).cpu().numpy().tolist() + [0]
wavelet_64_clcf = mexican_hat(31, torch.tensor(3.0), torch.tensor(0.2), False).cpu().numpy().tolist() + [0]
wavelet_32_clc_halfed = mexican_hat(31, torch.tensor(1.5), torch.tensor(0.2), False).cpu().numpy().tolist() + [0]

x = numpy.array(list(range(32))) - 15

fig, axs = plt.subplots(3, 1)

axs[0].bar(x, wavelet_64_clca, label="WV64CLCA")
axs[1].bar(x, wavelet_64_clcf, label="WV64CLCA")
axs[2].bar(x, wavelet_32_clc_halfed, label="WV64CLCA_halfed")

plt.legend()
plt.show()