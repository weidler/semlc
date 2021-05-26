import os

import torch
from matplotlib import pyplot as plt

from analysis.util import load_model_by_id
from config import CONFIG
from networks.util import build_network
from utilities.image import grid_plot, grayify_rgb_filters

id = "1622052621344469"

model = load_model_by_id(id, location_modifier="")
filters = model.get_conv_one().weight

torch.save(filters, os.path.join(CONFIG.PRETRAIN_DIR, "v1_pretraining.pt"))

lc_layer_function = model.lateral_layer_function
network = build_network("shallow", input_shape=model.input_shape, n_classes=10, lc=lc_layer_function)

grid_plot(grayify_rgb_filters(network.get_conv_one().weight.cpu().detach().numpy()))
plt.show()

network.init_gabors()
grid_plot(grayify_rgb_filters(network.get_conv_one().weight.cpu().detach().numpy()))
plt.show()

network.init_pretraining()
grid_plot(grayify_rgb_filters(network.get_conv_one().weight.cpu().detach().numpy()))
plt.show()

