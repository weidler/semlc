import sys

sys.path.append("../")

import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torchvision import transforms, datasets

import pandas as pd

from utilities.eval import accuracy, accuracy_from_data_loader
from networks import ConvergedInhibitionNetwork

#torch.random.manual_seed(12311)
#np.random.seed(12311)
#random.seed(12311)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

print(f"USE CUDA: {use_cuda}.")


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
    # use these for now as the baseline uses these
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # normalize = transforms.Normalize(
    #    mean=[0.4914, 0.4822, 0.4465],
    #    std=[0.2023, 0.1994, 0.2010],
    # )

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

    return train_loader, valid_loader, test_dataset, valid_dataset


def get_random_samples(samples, range_scope, range_ricker_width, range_damp):
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
    df = pd.read_csv("../data/hp_config.csv", dtype={'size': int, 'width': int, 'damp': float})
    configurations = df.values
    return configurations


if __name__ == "__main__":
    batch_size = 128
    _, valid_loader, test_set, valid_set = get_train_valid_loaders("../data/cifar10/", batch_size)
    strategies = ["converged", "toeplitz"]# "once", "once_learned"]
    # strategies = ["once"]
    # size is specific to each layers
    range_scope = np.array([[9, 27, 45, 63],
                            [9, 27, 45, 63],
                            [9, 27, 45, 63],
                            [7, 17, 25, 31],
                            ])
    range_ricker_width = [3, 4, 6, 8, 10]
    range_damp = [0.1, 0.12, 0.14, 0.16, 0.2]
    samples = 30
    # configurations = get_random_samples(samples, range_scope, range_ricker_width, range_damp)
    # configurations = [[27, 3, 0.1]]
    configurations = get_samples_from_disk()

    # df = pd.DataFrame(columns=["val_acc", "test_acc", "strategy", "size", "width", "damp"])

    for i in range(1,3):
        df = pd.DataFrame(columns=["val_acc", "test_acc", "strategy", "size", "width", "damp"])
        for strategy in strategies:
            for scope, ricker_width, damp in configurations:
                print("starting", f"str: {strategy} sc: {scope} w: {ricker_width} d: {damp}")
                # fix size when applying depth > 1
                net = ConvergedInhibitionNetwork()

                freeze='_freeze' if strategy=='toeplitz' else ''
                net.load_state_dict(torch.load(f"../saved_models/0{i}/{strategy}/scope_{scope}/width_{ricker_width}/damp_{damp}/ConvergedInhibitionNetwork{freeze}_final.model"))
                val_acc = accuracy_from_data_loader(net, valid_loader)
                test_acc = accuracy(net, test_set, batch_size=batch_size)
                df = df.append({'val_acc': val_acc, 'test_acc': test_acc, 'strategy': strategy, 'size': scope, 'width': ricker_width, 'damp': damp}, ignore_index=True)

                df = df.sort_values(by='val_acc', ascending=False)
                df.to_csv(path_or_buf=f"../results/0{i}/hpopt_0{i}.csv", index=False)
