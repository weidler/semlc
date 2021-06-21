import torch
from matplotlib import pyplot as plt

from core.weight_initialization import difference_of_gaussians

w1 = 7.515510791912675
w2 = 9.617999628186226
r = torch.tensor(0.2596823684871197)
d = torch.tensor(0.00855343253351748)

profile = difference_of_gaussians(63, (torch.tensor(w1), torch.tensor(w2)), r, d)
plt.bar(range(63), profile)
plt.show()
