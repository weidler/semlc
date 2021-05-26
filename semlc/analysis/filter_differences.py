"""Calculates the mean mse_differences between filters_per_group in certain groups and layers and writes the results to a file"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from analysis.plot_ordering import mse_difference, mae_difference
from utilities.eval import accuracies_from_list
from utilities.filter_ordering import mse, mae
from analysis.util import get_group_model_ids, load_model_by_id

# 30 for parametric else 60
num_layer = 0

groups = [
    'cornetz-cifar10-semlc',
    # 'cornetz-cifar10-parametric-semlc',
    'cornetz-cifar10-gaussian-semlc',
    'cornetz-cifar10',
    # 'alexnet-cifar10',
    # 'capsnet-mnist',
    # 'capsnet-mnist-semlc',
    # 'capsnet-mnist-lrn',
]

for group in groups:
    models = []
    for id in get_group_model_ids(group):
        try:
            model = load_model_by_id(id)
            models.append(model)
        except:
            pass

    mses, maes = [], []
    mses_neighbor, maes_neighbor = [], []

    for net in tqdm(models, disable=True):
        filters = net.conv_one.weight.data.numpy()
        scaler_mse = MinMaxScaler()
        scaler_mae = MinMaxScaler()

        mse_differences, mae_differences = [], []
        for i in range(-1, len(filters)):
            for j in range(i + 1, len(filters)):
                mse_diff = mse(filters[i], filters[j])
                mae_diff = mae(filters[i], filters[j])
                mse_differences.append(mse_diff)
                mae_differences.append(mae_diff)

        mse_differences = np.array(mse_differences).reshape(-1, 1)
        mse_differences = scaler_mse.fit_transform(mse_differences)[:, 0].tolist()

        mae_differences = np.array(mae_differences).reshape(-1, 1)
        mae_differences = scaler_mae.fit_transform(mae_differences)[:, 0].tolist()

        tmp_mse = accuracies_from_list(mse_differences, dec=3)
        tmp_mae = accuracies_from_list(mae_differences, dec=3)
        mses.append(tmp_mse[0])
        maes.append(tmp_mae[0])

        mse_neighbor = mse_difference(filters, scaler=scaler_mse)
        mae_neighbor = mae_difference(filters, scaler=scaler_mae)
        mses_neighbor.append(mse_neighbor)
        maes_neighbor.append(mae_neighbor)

    mean_mse = accuracies_from_list(mses, dec=8)
    mean_mae = accuracies_from_list(maes, dec=8)
    mean_neighboring_mse = accuracies_from_list(mses_neighbor, dec=8)
    mean_neighboring_mae = accuracies_from_list(maes_neighbor, dec=8)

    with open(f"diff_{group}_layer_{num_layer}.out", 'a') as f:
        f.write(f'mean MSE:\t{round(mean_mse[0], 3)}\n')
        f.write(f"mean adjacent MSE:\t{round(mean_neighboring_mse[0], 3)}\n")
        f.write(f"percent less chaos:\t {round((mean_mse[0] - mean_neighboring_mse[0]) / mean_mse[0], 3) * 100}")

        f.write(f'\n\nmean MAE:\t{round(mean_mae[0], 3)}\n')
        f.write(f"mean adjacent MAE:\t{round(mean_neighboring_mae[0], 3)}\n")
        f.write(f"percent less chaos:\t {round((mean_mae[0] - mean_neighboring_mae[0]) / mean_mae[0], 3) * 100}")

    print(f"Done {group}.")
