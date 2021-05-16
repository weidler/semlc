import json
import os
import re
from typing import List

import torch

from config import CONFIG

# files to saved models and keychain
from networks import BaseNetwork
from networks.util import build_network
from layers.util import prepare_lc_builder


# keychain = "./output/keychain.txt"
# path = "./output/"

# def get_net(group: str):
#     """
#     loads the layers with pre-defined hyper parameters for a given group
#
#     :param group:                the group
#
#     :return:                        the layers
#     """
#
#     all_nets = {
#         # baselines
#         'baseline': Baseline(),
#         'cmap': BaselineCMap(),
#
#         # ssi
#         'ss': SingleShotInhibitionNetwork(8, 0.2),
#         'ss_freeze': SingleShotInhibitionNetwork(3, 0.1),
#         'ss_freeze_zeros': SingleShotInhibitionNetwork(3, 0.1, pad="zeros"),
#         'ss_freeze_self': SingleShotInhibitionNetwork(3, 0.1, self_connection=True),
#         'ss_zeros': SingleShotInhibitionNetwork(8, 0.2, pad="zeros"),
#         'ss_self': SingleShotInhibitionNetwork(3, 0.1, self_connection=True),
#
#         # converged
#         'converged': ConvergedInhibitionNetwork(3, 0.1),
#         'converged_freeze': ConvergedInhibitionNetwork(3, 0.2),
#         'converged_zeros': ConvergedInhibitionNetwork(3, 0.1, pad="zeros"),
#         'converged_freeze_zeros': ConvergedInhibitionNetwork(3, 0.2, pad="zeros"),
#         'converged_self': ConvergedInhibitionNetwork(3, 0.1, self_connection=True),
#         'converged_freeze_self': ConvergedInhibitionNetwork(3, 0.2, self_connection=True),
#         'converged_cov_12': ConvergedInhibitionNetwork([3, 3], [0.1, 0.1]),
#         'converged_cov_123': ConvergedInhibitionNetwork([3, 3, 3], [0.1, 0.1, 0.1]),
#         'converged_full': ConvergedInhibitionNetwork([3, 3, 3, 3], [0.1, 0.1, 0.1, 0.1]),
#         'converged_full_best': ConvergedInhibitionNetwork([3, 10, 3, 10], [0.12, 0.1, 0.14, 0.12]),
#
#         # parametric
#         'parametric': ParametricInhibitionNetwork(3, 0.2),
#         'parametric_zeros': ParametricInhibitionNetwork(3, 0.2, pad="zeros"),
#         'parametric_self': ParametricInhibitionNetwork(3, 0.2, self_connection=True),
#         'parametric_12': ParametricInhibitionNetwork([3, 3], [0.2, 0.2]),
#         'parametric_123': ParametricInhibitionNetwork([3, 3, 3], [0.2, 0.2, 0.2]),
#
#         # vgg
#     }
#
#     return all_nets[group]
#
#
# def get_all_model_paths(group: str):
#     """
#     returns all file paths to saved models for a given group
#     :param group:            the group
#
#     :return:                    a list of file paths
#     """
#
#     files = df[df['group'].str.match(rf'{group}_\d\d?')]['id']
#     return files
#
#
# def get_one_model(group: str, index=0):
#     """
#     returns a layers with loaded state dictionary at the specified index of all saved models
#
#     :param group:            the group
#     :param index:               the index
#
#     :return:                    the layers with loaded state dictionary
#     """
#
#     model_path = get_all_model_paths(group).iloc[index]
#     filename = f"{path}{model_path}_best.layers"
#     model = get_net(group)
#     model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
#     return model


def get_group_model_ids(group) -> List[str]:
    group_names = [group]

    exp_dir = "../" + CONFIG.MODEL_DIR
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    ids = []
    for path in experiment_paths:
        eid_m = re.match("[0-9]+", str(path.split("/")[-1]))
        if eid_m:
            with open(os.path.join(path, "meta.json"), "r") as f:
                meta = json.load(f)
            group = meta.get("group")

            if group not in group_names:
                continue

            ids.append(eid_m.group(0))

    return ids


def load_model_by_id(model_id: str, location_modifier: str = "../") -> BaseNetwork:
    model_dir = os.path.join(location_modifier + CONFIG.MODEL_DIR, str(model_id))

    with open(f"{model_dir}/meta.json") as f:
        meta = json.load(f)

    # MAKE MODEL
    image_width, image_height = (meta.get("input_width"), meta.get("input_height"))
    n_classes = meta.get("dataset").get("n_classes")
    lc = prepare_lc_builder(meta.get("lateral_type"), 3, .2)
    model = build_network(meta["network_type"], input_shape=(meta["input_channels"], image_height, image_width),
                          n_classes=n_classes, lc=lc)

    model.load_state_dict(torch.load(f"{model_dir}/best.parameters"))

    return model
