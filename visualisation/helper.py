import pandas as pd
import torch

from networks import vgg19, vgg19_inhib
from networks import SingleShotInhibitionNetwork, BaselineCMap, Baseline, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork

# files to saved models and keychain
keychain = "./output/keychain.txt"
path = "./output/"

df = pd.read_csv(keychain, sep="\t", names=['id', 'group', 'layers', 'datetime'])


def get_net(strategy: str):
    """
    loads the layers with pre-defined hyper parameters for a given strategy

    :param strategy:                the strategy

    :return:                        the layers
    """

    all_nets = {
        # baselines
        'baseline': Baseline(),
        'cmap': BaselineCMap(),

        # ssi
        'ss': SingleShotInhibitionNetwork(8, 0.2),
        'ss_freeze': SingleShotInhibitionNetwork(3, 0.1),
        'ss_freeze_zeros': SingleShotInhibitionNetwork(3, 0.1, pad="zeros"),
        'ss_freeze_self': SingleShotInhibitionNetwork(3, 0.1, self_connection=True),
        'ss_zeros': SingleShotInhibitionNetwork(8, 0.2, pad="zeros"),
        'ss_self': SingleShotInhibitionNetwork(3, 0.1, self_connection=True),

        # converged
        'converged': ConvergedInhibitionNetwork(3, 0.1),
        'converged_freeze': ConvergedInhibitionNetwork(3, 0.2),
        'converged_zeros': ConvergedInhibitionNetwork(3, 0.1, pad="zeros"),
        'converged_freeze_zeros': ConvergedInhibitionNetwork(3, 0.2, pad="zeros"),
        'converged_self': ConvergedInhibitionNetwork(3, 0.1, self_connection=True),
        'converged_freeze_self': ConvergedInhibitionNetwork(3, 0.2, self_connection=True),
        'converged_cov_12': ConvergedInhibitionNetwork([3, 3], [0.1, 0.1]),
        'converged_cov_123': ConvergedInhibitionNetwork([3, 3, 3], [0.1, 0.1, 0.1]),
        'converged_full': ConvergedInhibitionNetwork([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1]),
        'converged_full_best': ConvergedInhibitionNetwork([3, 10, 3, 10], [0.12, 0.1, 0.14, 0.12]),

        # parametric
        'parametric': ParametricInhibitionNetwork(3, 0.2),
        'parametric_zeros': ParametricInhibitionNetwork(3, 0.2, pad="zeros"),
        'parametric_self': ParametricInhibitionNetwork(3, 0.2, self_connection=True),
        'parametric_12': ParametricInhibitionNetwork([3, 3], [0.2, 0.2]),
        'parametric_123': ParametricInhibitionNetwork([3, 3, 3], [0.2, 0.2, 0.2]),

        # vgg
        'vgg19': vgg19(),
        'vgg19_inhib': vgg19_inhib()
    }

    return all_nets[strategy]


def get_all_model_paths(strategy: str):
    """
    returns all file paths to saved models for a given strategy
    :param strategy:            the strategy

    :return:                    a list of file paths
    """

    files = df[df['group'].str.match(rf'{strategy}_\d\d?')]['id']
    return files


def get_one_model(strategy: str, index=0):
    """
    returns a layers with loaded state dictionary at the specified index of all saved models

    :param strategy:            the strategy
    :param index:               the index

    :return:                    the layers with loaded state dictionary
    """

    model_path = get_all_model_paths(strategy).iloc[index]
    filename = f"{path}{model_path}_best.layers"
    model = get_net(strategy)
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    return model
