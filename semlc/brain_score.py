import functools
from model_tools.activations.pytorch import load_preprocess_images, load_images
from model_tools.activations.pytorch import PytorchWrapper
from brainscore import score_model
from model_tools.brain_transformation import ModelCommitment

from semlc.networks.util import build_network
from semlc.layers.util import prepare_lc_builder
import torch
from torch import nn
import numpy as np


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
        self.relu1 = torch.nn.ReLU()
        linear_input_size = np.power((32 - 3 + 2 * 0) / 1 + 1, 2) * 2
        self.linear = torch.nn.Linear(int(linear_input_size), 10)
        self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.relu2(x)
        return x


lateral_connectivity_function = prepare_lc_builder("semlc", 3, 0.2)

# sh_net = build_network("shallow", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
# sb_net = build_network("simple", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)
alex_net = build_network("alexnet", input_shape=(3, 32, 32), n_classes=10, lc=lateral_connectivity_function)

preprocessing = functools.partial(load_images, image_filepaths='../data/mnist')

activations_model = PytorchWrapper(identifier='alexnet', model=alex_net, preprocessing=preprocessing)
model = ModelCommitment(identifier='alexnet', activations_model=activations_model,
                        # specify layers to consider
                        layers=['conv1'])
score = score_model(model_identifier=model.identifier, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)