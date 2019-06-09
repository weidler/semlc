import re

import torch

from torch import nn


class Logger:

    def __init__(self, model: nn.Module):
        self.model = model
        inhibition_strategy = self.model.inhibition_strategy if hasattr(self.model, 'inhibition_strategy') else ''
        self.loss_filename = f"../results/{model.__class__.__name__}_{inhibition_strategy}.loss"
        self.model_filename = f"../saved_models/{model.__class__.__name__}_{inhibition_strategy}_n.model"

        self.loss_history = []

    def update(self, loss, epoch):
        self.loss_history.append((epoch, loss))
        with open(self.loss_filename, "w") as f:
            f.write("\n".join([f"{e}\t{l}" for e, l in self.loss_history]))

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), re.sub("_n", f"_{epoch}", self.model_filename))
