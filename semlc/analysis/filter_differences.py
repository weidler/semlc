"""Calculates the mean mse_differences between filters_per_group in certain groups and layers and writes the results to a file"""
import statistics

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from analysis.plot_ordering import mse_difference, mae_difference
from utilities.eval import accuracies_from_list
from utilities.filter_ordering import mse, mae
from analysis.util import get_group_model_ids, load_model_by_id


def calc_order_statistics(filters: torch.Tensor):
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

    mse_all = accuracies_from_list(mse_differences, dec=3)
    mae_all = accuracies_from_list(mae_differences, dec=3)

    mse_neighbor = mse_difference(filters, scaler=scaler_mse)
    mae_neighbor = mae_difference(filters, scaler=scaler_mae)

    percent_less_chaos = (mse_all[0] - mse_neighbor) / mse_all[0]

    return mse_all[0], mae_all[0], mse_neighbor, mae_neighbor, percent_less_chaos


if __name__ == "__main__":
    # 30 for parametric else 60
    num_layer = 0

    groups = [
        'shallow-cifar10-semlc',
        # 'cornetz-cifar10-parametric-semlc',
        # 'cornetz-cifar10-gaussian-semlc',
        # 'cornetz-cifar10',
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
        percent_less_chaoses = []

        for net in tqdm(models, disable=True):
            filters = net.conv_one.weight.data.numpy()
            net_mse, net_mae, mse_neighbor, mae_neighbor, percent_less_chaos = calc_order_statistics(filters)
            mses.append(net_mse)
            maes.append(net_mae)
            mses_neighbor.append(mse_neighbor)
            maes_neighbor.append(mae_neighbor)
            percent_less_chaoses.append(percent_less_chaos)

        mean_mse = statistics.mean(mses)
        mean_mae = statistics.mean(maes)
        mean_neighboring_mse = statistics.mean(mses_neighbor)
        mean_neighboring_mae = statistics.mean(maes_neighbor)
        percent_less_chaos = statistics.mean(percent_less_chaoses)

        with open(f"diff_{group}_layer_{num_layer}-f.out", 'w') as f:
            f.write(f'mean MAE:\t{round(mean_mae, 3)}\n')
            f.write(f"mean adjacent MAE:\t{round(mean_neighboring_mae, 3)}\n")

            f.write(f'mean MSE:\t{round(mean_mse, 3)}\n')
            f.write(f"mean adjacent MSE:\t{round(mean_neighboring_mse, 3)}\n")
            f.write(f"percent less chaos:\t {round(percent_less_chaos, 3) * 100}")

        print(f"Done {group}.")
