import time

import numpy
import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.statistics import acc
from networks import BaseNetwork
from utilities.evaluation import evaluate_classification
from utilities.log import ExperimentLogger


def train_model(model: BaseNetwork,train_set_loader: DataLoader, val_set_loader: DataLoader,
                n_epochs: int, device: torch.device, logger: ExperimentLogger):
    """Train a given model on a given dataset with the provided settings.

    Args:
        model: network instance that will be trained, can be instance of any class inheriting from BaseNetwork
        optimizer:
        lr_scheduler:
        train_set_loader:
        val_set_loader:
        n_epochs:
        device:
        logger:
    """

    model.train()
    criterion = model.__class__.make_preferred_criterion()
    optimizer = model.make_preferred_optimizer()
    lr_scheduler = model.make_preferred_lr_schedule(optimizer)

    best_performance = - numpy.inf

    total_batches = 0
    train_start_time = time.time()
    for epoch in range(1, n_epochs + 1):

        total_epoch_loss = 0
        episode_mbatches_seen = 0

        epoch_start_time = time.time()
        for i, data in tqdm(enumerate(train_set_loader, 0), total=len(train_set_loader), disable=False,
                                 desc=f"Training Epoch {epoch} of {n_epochs}", leave=False):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            total_epoch_loss += loss.item()
            episode_mbatches_seen += 1

            loss.backward()
            optimizer.step()

        train_loss = total_epoch_loss / episode_mbatches_seen
        total_batches += episode_mbatches_seen

        model.eval()
        with torch.no_grad():
            correct, total, val_loss = evaluate_classification(model, val_set_loader, criterion=criterion,
                                                               device=device)

            performance = acc(correct.sum(), total.sum()).item()
            is_new_best = performance > best_performance
            performance_string = f"Validation Accuracy: {'*' if is_new_best else ' '}{round(performance, 3):7.3f}"

        if is_new_best:
            best_performance = performance
            torch.save(model.state_dict(), f"{logger.model_dir}/best.parameters")

        print(f"[EPOCH {epoch:3d}] {performance_string}; "
              f"Current LR: {round(optimizer.param_groups[0].get('lr'), 5):7.5f}; "
              f"Train Loss: {round(train_loss, 4):6.4f}; "
              f"Val Loss: {round(val_loss, 4):6.4f}; "
              f"Total Batches Seen: {total_batches:5d}; "
              f"Time Taken: {round(time.time() - epoch_start_time, 2):6.2f}s; "
              f"Time Left: ~{(n_epochs - epoch - 1) * round((time.time() - epoch_start_time) / 60, 2):6.2f}m")

        logger.log(epoch, train_loss, val_loss, performance)

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

    # finalize training by saving ultimate parameters and adjusting some meta
    logger.finalize(round(time.time() - train_start_time, 2))


def adjust_learning_rate(l_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = l_rate * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    from networks import InhibitionNetwork, BaseNetwork, Optimizer

    model = InhibitionNetwork()
