import sys

import pandas as pd

from util.train import train_model

sys.path.append("./")

import random
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torchvision import transforms, datasets

from util.eval import accuracy
from util.ourlogging import Logger
from model.network.alexnet_cifar import ConvergedInhibitionNetwork

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")

strategies = ["converged", "toeplitz", "once", "once_learned"]
# scope is specific to each layer
range_scope = np.array([[9, 27, 45, 63],
                        [9, 27, 45, 63],
                        [9, 27, 45, 63],
                        [7, 17, 25, 31],
                        ])
range_ricker_width = [3, 4, 6, 8, 10]
range_damp = [0.1, 0.12, 0.14, 0.16, 0.2]


def get_train_valid_loaders(data_dir, batch_size, augment=True, valid_size=0.2, shuffle=True, pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        valid_transform = train_transform = transforms.Compose([
            transforms.RandomCrop(24),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_dataset


def get_random_samples(samples, range_scope, range_ricker_width, range_damp):
    """
    creates a number of unique random configurations

    :param samples:                 the number of samples
    :param range_scope:             an array containing all considered values for the scope
    :param range_ricker_width:      an array containing all considered values for the ricker width
    :param range_damp:              an array containing all considered values for the damping factor

    :return:                        a list of configurations
    """
    configurations = []
    for i in range(samples):
        # for the lack of do while loops
        scope = random.choice(range_scope.T)
        width = random.choice(range_ricker_width)
        damp = random.choice(range_damp)
        config = [scope[0], width, damp]
        while configurations.count(config) > 0:
            scope = random.choice(range_scope.T)
            width = random.choice(range_ricker_width)
            damp = random.choice(range_damp)
            config = [scope[0], width, damp]
        configurations.append(config)
    return configurations


def get_samples_from_disk():
    """
    laods already generated configurations from disk

    :return:        a list of configurations
    """
    df = pd.read_csv("./data/hp_config.csv", dtype={'scope': int, 'width': int, 'damp': float})
    configurations = df.values
    return configurations


def hp_opt(num_epoch, train_loader, val_loader, criterion, learn_rate=0.01, test_set=None,
           optimizer=None, verbose=True):
    """
    runs a hyper parameter optimisation with hyper parameter sets loaded from disk

    :param num_epoch:           the number of epochs
    :param train_loader:        the training data loader
    :param val_loader:          the validation data loader
    :param criterion:           the loss criterion
    :param learn_rate:          the learning rate
    :param test_set:            the test set
    :param optimizer:           the optimizer
    :param verbose:             whether to print updates during running the optimisation

    """
    configurations = get_samples_from_disk()
    for strategy in strategies:
        for scope, ricker_width, damp in configurations:
            print("starting",
                  f"str: {strategy} freeze: {strategy == 'toeplitz'} sc: {int(scope)} w: {int(ricker_width)} d: {damp}")
            # fix scope when applying depth > 1
            net = ConvergedInhibitionNetwork()

            if use_cuda:
                net.cuda()

            logger = Logger(net)

            # Adam optimizer by default
            if optimizer is None:
                optimizer = optim.Adam(net.parameters(), lr=learn_rate)

            train_model(net=net,
                        num_epoch=num_epoch,
                        train_loader=train_loader,
                        criterion=criterion,
                        logger=logger,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        learn_rate=learn_rate,
                        verbose=verbose)

            end_acc = accuracy(net, test_set, batch_size)
            logger.log(f"test acc: {end_acc}")
            optimizer = None


if __name__ == "__main__":
    batch_size = 128
    l_rate = 0.001
    train_loader, valid_loader, test_set = get_train_valid_loaders("./data/cifar10/", batch_size)
    hp_opt(num_epoch=1,
           train_loader=train_loader,
           val_loader=valid_loader,
           criterion=nn.CrossEntropyLoss(),
           learn_rate=l_rate,
           test_set=test_set,
           verbose=True)
