import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from torchvision import transforms, datasets

from experiment.eval import accuracy
from experiment.train import custom_optimizer_conv11
from util.ourlogging import Logger
from model.network.alexnet_paper import ConvNet11


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
            transforms.CenterCrop(24),
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
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

    return train_loader, valid_loader


def hp_opt(num_epoch, train_loader, val_loader, criterion, learn_rate=0.01, test_set=None, optimizer=None):
    strategies = ["once", "converged", "toeplitz"]
    # scope is specific to each layer
    range_scope = np.array([[17, 19, 21, 23],
                            [33, 35, 37, 39],
                            [33, 35, 37, 39],
                            [33, 35, 37, 39],
                            ])
    range_ricker_width = np.linspace(3.0, 6.0, num=4)
    range_damp = np.linspace(0.1, 0.16, num=4)
    best_loss = np.Infinity
    best_net = None
    test = 0
    for scope in range_scope.T:
        for ricker_width in range_ricker_width:
            for damp in range_damp:
                for strategy in strategies:
                    net = ConvNet11(scope=scope,
                                    width=ricker_width,
                                    damp=damp,
                                    inhibition_depth=2,
                                    inhibition_strategy=strategy,
                                    logdir=f"scope_{scope[0]}/width_{ricker_width}/damp_{damp}/{strategy}"
                                    )
                    logger = Logger(net)
                    
                    # Adam optimizer by default
                    if optimizer is None:
                        optimizer = optim.Adam(net.parameters(), lr=learn_rate)

                    loss_history = []
                    num_examples = train_loader.__len__()
                    for epoch in range(num_epoch):  # loop over the dataset multiple times
                        running_loss = 0.0
                        if epoch == 100:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = param_group['lr'] * 0.1

                        for i, (inputs, labels) in enumerate(train_loader, 0):
                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward + backward + optimize
                            if torch.cuda.is_available():
                                outputs = net(inputs.cuda())
                            else:
                                outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                            # print statistics
                            running_loss += loss.item()
                            if i == num_examples - 1:
                                log_loss = running_loss / num_examples
                                loss_history.append(log_loss)
                                if logger is not None:
                                    logger.update_loss(log_loss, epoch + 1)
                                    if test_set is not None:
                                        acc = accuracy(net, test_set, batch_size)
                                        logger.log('[%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1, log_loss, acc),
                                                   console=True)
                                        logger.update_acc(acc, epoch + 1)
                                    else:
                                        logger.log('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, log_loss), console=True)
                                running_loss = 0.0

                        if epoch % 5 == 0 or epoch == num_epoch - 1:
                            logger.save_model(epoch)
                            logger.save_optimizer(optimizer, epoch)

                    model_loss = 0.0
                    # check validation loss to decide which model we keep
                    for i, (inputs, labels) in enumerate(val_loader, 0):
                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        if torch.cuda.is_available():
                            outputs = net(inputs.cuda())
                        else:
                            outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        model_loss += loss.item()
                    if best_loss > model_loss:
                        best_loss = model_loss
                        # save the model
                        logger.save_model(num_epoch, best=True)
                        best_net = net

    return best_net


if __name__ == "__main__":
    batch_size = 128
    l_rate = 0.001
    train_loader, valid_loader = get_train_valid_loaders("../data/cifar10/", batch_size)
    hp_opt(num_epoch=10,
           train_loader=train_loader,
           val_loader=valid_loader,
           criterion=nn.CrossEntropyLoss(),
           learn_rate=0.001
           )

