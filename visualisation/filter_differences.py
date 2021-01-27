"""Calculates the mean differences between filters in certain strategies and layers and writes the results to a file"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from utilities.eval import accuracies_from_list
from utilities.filter_ordering import mse
from visualisation.util import get_group_model_ids, load_model_by_id
from visualisation.plot_ordering import mse_difference

# 30 for parametric else 60
num_layer = 0

groups = [
    'alexnet-cifar10-semlc',
    'alexnet-cifar10-adaptive-semlc',
    'alexnet-cifar10-parametric-semlc',
    'alexnet-cifar10-gaussian-semlc',
    'alexnet-cifar10-lrn',
    'alexnet-cifar10',
]

for group in groups:
    models = []
    for id in get_group_model_ids(group):
        model = load_model_by_id(id)
        models.append(model)

    mses = []
    mses_neighbor = []
    confidences = []

    for net in tqdm(models, disable=True):
        filters = net.conv_one.weight.data.numpy()
        scaler = MinMaxScaler()

        differences = []
        for i in range(-1, len(filters)):
            for j in range(i + 1, len(filters)):
                diff = mse(filters[i], filters[j])
                differences.append(diff)

        differences = np.array(differences).reshape(-1, 1)
        differences = scaler.fit_transform(differences)[:, 0].tolist()

        tmp = accuracies_from_list(differences, dec=3)
        mses.append(tmp[0])
        confidences.append(tmp[:2])

        mse_neighbor = mse_difference(filters, scaler=scaler)
        mses_neighbor.append(mse_neighbor)

    all_acc = accuracies_from_list(mses, dec=3)
    neighboring = accuracies_from_list(mses_neighbor, dec=3)

    with open(f"diff_{group}_layer_{num_layer}.out", 'a') as f:
        for m in confidences:
            f.write(f"{m[0]}\t{m[1]}\n")
        f.write(f'mean:\t{all_acc}\n')
        f.write(f"mean neighboring:\t{neighboring}\n")
        f.write(f"percent less chaos:\t {round((all_acc[0] - neighboring[0]) / all_acc[0], 3) * 100}")

    print(f"Done {group}.")
