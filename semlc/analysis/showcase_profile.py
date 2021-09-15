import torch
from matplotlib import pyplot as plt

from core.weight_initialization import difference_of_gaussians

# shallow
w1 = 7.79
w2 = 9.96
r = torch.tensor(.84)
d = torch.tensor(0.017)

profile = difference_of_gaussians(63, (torch.tensor(w1), torch.tensor(w2)), r, d)
plt.bar(range(63), profile)
plt.show()
