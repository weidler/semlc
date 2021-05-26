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

    lc = prepare_lc_builder(meta.get("lateral_type"),
                            meta.get("lateral_layer").get("widths") if meta.get("is_lateral") else (3, 5),
                            meta.get("lateral_layer").get("ratio") if meta.get("is_lateral") else 2,
                            meta.get("lateral_layer").get("damping") if meta.get("is_lateral") else .2,
                            meta.get("lateral_layer").get("rings") if meta.get("is_lateral") else 1)

    model = build_network(meta["network_type"], input_shape=(meta["input_channels"], image_height, image_width),
                          n_classes=n_classes, lc=lc)

    model.load_state_dict(torch.load(f"{model_dir}/best.parameters"))

    return model
