import statistics
from typing import List
import numpy as np
import csv

import matplotlib.pyplot as plt
import torch
from matplotlib import gridspec
from matplotlib.axes import Axes
from torch import Tensor
from tqdm import tqdm

from util.eval import accuracies_from_list
from util.filter_ordering import two_opt, mse
from visualisation.filter_weights_visualization import get_ordering_difference, get_dim_for_plot
from visualisation.helper import get_one_model, get_net
from visualisation.plot_ordering import get_orderings, create_plot, mse_difference

num_nets = 60
num_layer = 13

# strategies = ['baseline', 'cmap', 'ss_freeze', 'ss', 'converged_freeze', 'converged', 'parametric']
strategies = ['converged_full_best' for _ in range(num_nets)]
# strategies = ['baseline', 'cmap', 'ss_freeze', 'ss']
# strategies = ['converged_freeze', 'converged', 'parametric']
# names = ["Baseline AlexNet", "Baseline AlexCMap", "Single Shot Frozen", "Single Shot Adaptive", "Converged Frozen", "Converged Adaptive", "Converged Parametric"]
# names = ["Baseline AlexNet", "Baseline AlexCMap", "Single Shot Frozen", "Single Shot Adaptive"]
# names = ["Converged Frozen", "Converged Adaptive", "Converged Parametric"]
names = ["AlexNet" for _ in range(num_nets)]
models = []

for i, strategy in enumerate(strategies):
    # get first model found
    model = get_one_model(strategy, index=i)  # 56 if strategy == 'converged_freeze' else 0)
    models.append(model)

print(models[0].features[num_layer])
'''
freeze = get_net("ss_freeze").load_state_dict(torch.load(
                # f"../final_results/{strategy}/{strategy}_{j}/ConvergedInhibitionNetwork_best.model",
                # f"../final_results/{strategy}_models/{strategy}_{j}/SingleShotInhibitionNetwork_freeze_best.model",
                f"../_deprecated/final_results/ss_freeze_models/ss_freeze_2/SingleShotInhibitionNetwork_freeze_final.model",
                map_location=lambda storage, loc: storage))
models.append(freeze)
'''

mses = []
mses_neighbor = []
confs = []
from sklearn.preprocessing import MinMaxScaler


for net in tqdm(models, disable=True):
    filters = net.features[num_layer].weight.data.numpy()
    scalers = {}
    '''
    for idx in range(filters.shape[0]):
        # for channel in range(filters.shape[1]):
        scalers[idx] = MinMaxScaler()
        filters[idx] = scalers[idx].fit_transform(filters[idx].reshape(-1, 1)).reshape(filters[idx].shape)
    '''

    # scaled_filters = scaler.fit_transform(filters.reshape(-1, 1))
    # scaled_filters = scaled_filters.reshape(filters.shape)

    scaler = MinMaxScaler()

    differences = []
    for i in range(-1, len(filters)):
        for j in range(i + 1, len(filters)):
            diff = mse(filters[i], filters[j])
            differences.append(diff)
            # if i == 0:
                # print(i, j, diff)
    # print(differences)
    differences = np.array(differences).reshape(-1, 1)
    differences = scaler.fit_transform(differences)[:, 0].tolist()
    # print('scaled', differences)
    tmp = accuracies_from_list(differences, dec=8)
    mses.append(tmp[0])
    confs.append(tmp[:2])

    mse_neighbor = mse_difference(filters, scaler=scaler)
    # print(mse_neighbor)
    mses_neighbor.append(mse_neighbor)


print('mse: ', sum(mses) / len(mses))
print('mean: ', statistics.mean(mses))
print('std: ', statistics.stdev(mses))
print('mses: ', mses)
all = accuracies_from_list(mses, dec=8)
print('confs: ', confs)
print('conf: ', all)
print('neighboring: ', mses_neighbor)
neighboring = accuracies_from_list(mses_neighbor, dec=8)
print('mean neighboring: ', neighboring)

with open(f"diff_{strategies[0]}_layer_{num_layer}.out", 'a') as f:
    for m in confs:
        f.write(f"{m[0]}\t{m[1]}\n")
    f.write(f'mean:\t{all}\n')

    f.write(f"mean neighboring:\t{neighboring}")

