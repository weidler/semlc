import math
from random import shuffle

from skimage.filters import gabor_kernel

import matplotlib.pyplot as plt
from skimage.transform import resize

from util.filter_ordering import two_opt
from util.filter_ordering import greedy

filters = []

for orientation in range(0, 180, 180//12):
    filters.append(gabor_kernel(0.2, math.radians(orientation)).real)

fig, axs = plt.subplots(3, 4)
fig2, axs2 = plt.subplots(3, 4)

shuffle(filters)
max_size = max([f.shape[0] for f in filters])
filters = [resize(f, (max_size, max_size)) for f in filters]
filters2 = filters

filters = two_opt(filters)
filters2 = greedy(filters2)

f = 0
for row in range(3):
    for col in range(4):
        axs[row, col].imshow(filters[f], cmap="gray")
        f += 1

plt.show()