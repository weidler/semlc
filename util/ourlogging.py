import datetime
import random
import re
import os
import torch
import time


from torch import nn

from model.deprecated_inhibition_layer import Conv3DSingleShotInhibition
from model.network.alexnet_paper import ConvergedInhibitionNetwork, SingleShotInhibitionNetwork, Baseline


class Logger:

    def __init__(self, model: nn.Module):
        self.model = model
        self.process_id = str(int(time.time() * 10000)) + str(random.randint(100000, 999999))
        self.loss_filename = f"../output/{self.process_id}.loss"
        self.acc_filename = f"../output/{self.process_id}.acc"
        self.log_filename = f"../output/{self.process_id}.log"
        self.model_filename = f"../output/{self.process_id}_n.model"
        self.best_model_filename = f"../output/{self.process_id}_best.model"

        with open("../output/keychain.txt", "a") as f:
            f.write(
                f"{self.process_id}\t{repr(model)}\t{datetime.datetime.now()}\n"
            )

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
        path = re.sub("_n", f"_{epoch}", self.model_filename if not best else self.best_model_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def log(self, data, console=False):
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "a") as f:
            f.write(f"{data}\n")
        if console:
            print(data)

    def describe_network(self):
        print(repr(self.model))


if __name__ == "__main__":
    net = SingleShotInhibitionNetwork(scopes=[9],
                                     width=int(4),
                                     damp=0.1,
                                     freeze=True)
    base = Baseline()

    logger = Logger(base)
    print(logger.process_id)
