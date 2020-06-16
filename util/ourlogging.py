import datetime
import random
import re
import os
import torch
import time


from torch import nn

from model.alternative_inhibition_layers import Conv3DSingleShotInhibition
from model.network.alexnet_cifar import ConvergedInhibitionNetwork, SingleShotInhibitionNetwork, Baseline


class Logger:
    """A custom logger for all experiments"""

    def __init__(self, model: nn.Module, experiment_code: str = ""):
        self.model = model
        self.process_id = str(int(time.time() * 10000)) + str(random.randint(100000, 999999))
        self.loss_filename = f"./output/{self.process_id}.loss"
        self.acc_filename = f"./output/{self.process_id}.acc"
        self.log_filename = f"./output/{self.process_id}.log"
        self.model_filename = f"./output/{self.process_id}_n.model"
        self.best_model_filename = f"./output/{self.process_id}_best.model"
        self.opt_filename = f"./output/{self.process_id}_n.opt"
        self.best_opt_filename = f"./output/{self.process_id}_best.opt"

        with open("./output/keychain.txt", "a") as f:
            f.write(
                f"{self.process_id}\t{experiment_code}\t{repr(model)}\t{datetime.datetime.now()}\n"
            )

        self.loss_history = []
        self.acc_history = []

    def update_loss(self, loss, epoch):
        """
        updates the loss log file with the new loss

        :param loss:            the loss
        :param epoch:           the epoch

        """
        self.loss_history.append((epoch, loss))
        os.makedirs(os.path.dirname(self.loss_filename), exist_ok=True)
        with open(self.loss_filename, "w") as f:
            f.write("\n".join([f"{e}\t{l}" for e, l in self.loss_history]))

    def update_acc(self, acc, epoch):
        """
        updates the accuracy log file with the new accuracy

        :param acc:             the accuracy
        :param epoch:           the epoch

        """
        self.acc_history.append((epoch, acc))
        os.makedirs(os.path.dirname(self.acc_filename), exist_ok=True)
        with open(self.acc_filename, "w") as f:
            f.write("\n".join([f"{e}\t{a}" for e, a in self.acc_history]))

    def save_model(self, epoch, best=False):
        """
        saves the current model to the disk

        :param epoch:           the epoch
        :param best:            whether the model is the currently best model or not

        """
        path = re.sub("_n", f"_{epoch}", self.model_filename if not best else self.best_model_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def log(self, data, console=False):
        """
        updates the log file with the given data

        :param data:            the data to log
        :param console:         whether to additionally print to the console or not

        """
        os.makedirs(os.path.dirname(self.log_filename), exist_ok=True)
        with open(self.log_filename, "a") as f:
            f.write(f"{data}\n")
        if console:
            print(data)

    def save_optimizer(self, optimizer, epoch, best=False):
        """
        saves the current state of the optimizer

        :param optimizer:           the optimizer
        :param epoch:               the epoch
        :param best:                whether the model is the currently best model or not

        """
        path = re.sub("_n", f"_{epoch}", self.opt_filename if not best else self.best_opt_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(optimizer.state_dict(), path)

    def describe_network(self):
        """prints information about the model"""

        print(repr(self.model))


if __name__ == "__main__":
    net = SingleShotInhibitionNetwork()
    base = Baseline()

    logger = Logger(base)
    print(logger.process_id)
