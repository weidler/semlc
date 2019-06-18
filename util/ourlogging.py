import re
import os
import torch

from torch import nn


class Logger:

    def __init__(self, model: nn.Module):
        self.model = model
        inhibition_strategy = '_' + self.model.inhibition_strategy if hasattr(self.model, 'inhibition_strategy') else ''
        logdir = self.model.logdir +'/' if hasattr(self.model, 'logdir') else ''
        self.loss_filename = f"../results/{logdir}{model.__class__.__name__}{inhibition_strategy}.loss"
        self.model_filename = f"../saved_models/{logdir}{model.__class__.__name__}{inhibition_strategy}_n.model"

        self.loss_history = []

    def update(self, loss, epoch):
        self.loss_history.append((epoch, loss))
        os.makedirs(os.path.dirname(self.loss_filename), exist_ok=True)
        with open(self.loss_filename, "w") as f:
            f.write("\n".join([f"{e}\t{l}" for e, l in self.loss_history]))

    def save_model(self, epoch):
        os.makedirs(os.path.dirname(self.model_filename), exist_ok=True)
        torch.save(self.model.state_dict(), re.sub("_n", f"_{epoch}", self.model_filename))
