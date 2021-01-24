"""calculates the mean differences between filters in certain strategies and layers and writes the results to a file"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from utilities.eval import accuracies_from_list
from utilities.filter_ordering import mse
from visualisation.helper import get_one_model
from visualisation.plot_ordering import mse_difference

# 30 for parametric else 60
num_nets = 60
num_layer = 0

strategy = 'converged'

models = []

for i in range(num_nets):
    model = get_one_model(strategy, index=i)
    models.append(model)

mses = []
mses_neighbor = []
confidences = []

for net in tqdm(models, disable=True):
    filters = net.features[num_layer].weight.data.numpy()
    scaler = MinMaxScaler()

    differences = []
    for i in range(-1, len(filters)):
        for j in range(i + 1, len(filters)):
            diff = mse(filters[i], filters[j])
            differences.append(diff)

    differences = np.array(differences).reshape(-1, 1)
    differences = scaler.fit_transform(differences)[:, 0].tolist()

    tmp = accuracies_from_list(differences, dec=8)
    mses.append(tmp[0])
    confidences.append(tmp[:2])

    mse_neighbor = mse_difference(filters, scaler=scaler)
    mses_neighbor.append(mse_neighbor)

all_acc = accuracies_from_list(mses, dec=8)
neighboring = accuracies_from_list(mses_neighbor, dec=8)

with open(f"diff_{strategy}_layer_{num_layer}.out", 'a') as f:
    for m in confidences:
        f.write(f"{m[0]}\t{m[1]}\n")
    f.write(f'mean:\t{all_acc}\n')

    f.write(f"mean neighboring:\t{neighboring}")

