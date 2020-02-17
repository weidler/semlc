import pandas as pd
import torch

from model.network.VGG import vgg19, vgg19_inhib
from model.network.alexnet_cifar import SingleShotInhibitionNetwork, BaselineCMap, Baseline, ConvergedInhibitionNetwork, \
    ParametricInhibitionNetwork

# files to saved models and keychain
keychain = "../../../experiments/all/keychain.txt"
path = "../../../experiments/all/"

df = pd.read_csv(keychain, sep="\t", names=['id', 'group', 'model', 'datetime'])


def get_net(strategy: str):
    """
    loads the model with pre-defined hyper parameters for a given strategy

    :param strategy:                the strategy

    :return:                        the model
    """

    all_nets = {
        # baselines
        'baseline': Baseline(),
        'cmap': BaselineCMap(),

        # ssi
        'ss': SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False),
        'ss_freeze': SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True),
        'ss_freeze_zeros': SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True, pad="zeros"),
        'ss_freeze_self': SingleShotInhibitionNetwork([27], 3, 0.1, freeze=True, self_connection=True),
        'ss_zeros': SingleShotInhibitionNetwork([63], 8, 0.2, freeze=False, pad="zeros"),
        'ss_self': SingleShotInhibitionNetwork([63], 3, 0.1, freeze=False, self_connection=True),

        # converged
        'converged': ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False),
        'converged_freeze': ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True),
        'converged_zeros': ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, pad="zeros"),
        'converged_freeze_zeros': ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True, pad="zeros"),
        'converged_self': ConvergedInhibitionNetwork([27], 3, 0.1, freeze=False, self_connection=True),
        'converged_freeze_self': ConvergedInhibitionNetwork([45], 3, 0.2, freeze=True, self_connection=True),
        'converged_cov_12': ConvergedInhibitionNetwork([27, 27], [3, 3], [0.1, 0.1], freeze=False, coverage=2),
        'converged_cov_123': ConvergedInhibitionNetwork([27, 27, 27], [3, 3, 3], [0.1, 0.1, 0.1], freeze=False,
                                                        coverage=3),
        'converged_full': ConvergedInhibitionNetwork([27, 27, 27, 27], [3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1], freeze=False,
                                                     coverage=4),
        'converged_full_best': ConvergedInhibitionNetwork([27, 63, 45, 31], [3, 10, 3, 10], [0.12, 0.1, 0.14, 0.12],
                                                          freeze=False, coverage=4),

        # parametric
        'parametric': ParametricInhibitionNetwork([45], 3, 0.2),
        'parametric_zeros': ParametricInhibitionNetwork([45], 3, 0.2, pad="zeros"),
        'parametric_self': ParametricInhibitionNetwork([45], 3, 0.2, self_connection=True),
        'parametric_12': ParametricInhibitionNetwork([63, 63], [3, 3], [0.2, 0.2], coverage=2),
        'parametric_123': ParametricInhibitionNetwork([63, 63, 63], [3, 3, 3], [0.2, 0.2, 0.2], coverage=3),

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
    returns a model with loaded state dictionary at the specified index of all saved models

    :param strategy:            the strategy
    :param index:               the index

    :return:                    the model with loaded state dictionary
    """

    model_path = get_all_model_paths(strategy).iloc[index]
    filename = f"{path}{model_path}_best.model"
    model = get_net(strategy)
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    return model
