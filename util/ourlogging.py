import re
import os
import torch

from torch import nn


class Logger:

    def __init__(self, model: nn.Module):
        self.model = model
        inhibition_strategy = '_' + self.model.inhibition_strategy if hasattr(self.model, 'inhibition_strategy') else ''
        self.logdir = f"{self.model.logdir}/" if hasattr(self.model, 'logdir') and self.model.logdir is not None else ''
        self.loss_filename = f"../results/{self.logdir}{model.__class__.__name__}{inhibition_strategy}.loss"
        self.acc_filename = f"../results/{self.logdir}{model.__class__.__name__}{inhibition_strategy}.acc"
        self.log_filename = f"../logs/{self.logdir}{model.__class__.__name__}{inhibition_strategy}.log"
        self.model_filename = f"../saved_models/{self.logdir}{model.__class__.__name__}{inhibition_strategy}_n.model"
        self.opt_filename = f"../saved_models/opt/{self.logdir}{model.__class__.__name__}{inhibition_strategy}_n.opt"

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

    def save_model(self, epoch, best=False):
        if not best:
            path = re.sub("_n", f"_{epoch}", self.model_filename)
        else:
            # hard coded length 16 of "../saved_models/"
            path = self.model_filename[:16] + "best/" + self.model_filename[16:]
            path = re.sub("_n", f"_{epoch}", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_optimizer(self, optimizer, epoch):
        os.makedirs(os.path.dirname(self.opt_filename), exist_ok=True)
        torch.save(optimizer.state_dict(), re.sub("_n", f"_{epoch}", self.opt_filename))

    def log(self, data, console=False):
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "a") as f:
            f.write(f"{data}\n")
        if console:
            print(data)
