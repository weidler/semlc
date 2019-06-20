import statistics

import torchvision
from typing import List, Tuple

import torch
from scipy.stats import sem, t
from torch import nn
from torch.utils.data import DataLoader, Dataset


def accuracy(net, data_set, batch_size):
    net.eval()
    data_loader = DataLoader(data_set, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            if torch.cuda.is_available():
                outputs = net(images.cuda())
            else:
                outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.train()
    return 100 * correct / total


def accuracy_with_confidence(networks: List[nn.Module], data: Dataset, batchsize: int, confidence: float=0.95) \
        -> Tuple[List[float], Tuple[float, float]]:
    """Determine the mean accuracy of a given list of networks, alongside the confidence interval of this mean.
    This way, for multiple training runs with random initialization of on architecture, the resulting networks can be
    evaluated for accuracy with more confidence about the true power of the architecture.

    :param networks:        list of network modules
    :param data:            test data set
    :param batchsize:       batchsize for testruns
    :param confidence:      confidence that mean lies in interval, given at range [0, 1]

    :return:                mean accuracy and confidence interval
    """

    accuracies = []
    for network in networks:
        acc = accuracy(network, data, batchsize)
        accuracies.append(acc)

    mean = statistics.mean(accuracies)
    error = sem(accuracies)
    h = error * t.ppf((1 + confidence) / 2., len(accuracies) - 1)
    interval = (mean - h, mean + h)

    return mean, interval
