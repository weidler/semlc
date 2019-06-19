import re
import os
import torch

from torch import nn


class Logger:

    def __init__(self, model: nn.Module):
        self.model = model
        inhibition_strategy = '_' + self.model.inhibition_strategy if hasattr(self.model, 'inhibition_strategy') else ''
        logdir = f"{self.model.logdir}/" if hasattr(self.model, 'logdir') and self.model.logdir is not None else ''
        self.loss_filename = f"../results/{logdir}{model.__class__.__name__}{inhibition_strategy}.loss"
        self.acc_filename = f"../results/{logdir}{model.__class__.__name__}{inhibition_strategy}.acc"
        self.log_filename = f"../logs/{logdir}{model.__class__.__name__}{inhibition_strategy}.log"
        self.model_filename = f"../saved_models/{logdir}{model.__class__.__name__}{inhibition_strategy}_n.model"
        self.opt_filename = f"../saved_models/opt/{logdir}{model.__class__.__name__}{inhibition_strategy}_n.opt"

        self.loss_history = []
        self.acc_history = []

    def update_loss(self, loss, epoch):
        self.loss_history.append((epoch, loss))
        os.makedirs(os.path.dirname(self.loss_filename), exist_ok=True)
        with open(self.loss_filename, "w") as f:
            f.write("\n".join([f"{e}\t{l}" for e, l in self.loss_history]))

    def update_acc(self, acc, epoch):
        self.acc_history.append((epoch, acc))
        os.makedirs(os.path.dirname(self.acc_filename), exist_ok=True)
        with open(self.acc_filename, "w") as f:
            f.write("\n".join([f"{e}\t{a}" for e, a in self.acc_history]))

    def save_model(self, epoch):
        os.makedirs(os.path.dirname(self.model_filename), exist_ok=True)
        torch.save(self.model.state_dict(), re.sub("_n", f"_{epoch}", self.model_filename))

    def save_optimizer(self, optimizer, epoch):
        os.makedirs(os.path.dirname(self.opt_filename), exist_ok=True)
        torch.save(optimizer.state_dict(), re.sub("_n", f"_{epoch}", self.opt_filename))

    def log(self, data, console=False):
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "a") as f:
            f.write(f"{data}\n")
        if console:
            print(data)
