from typing import List

import numpy
import scipy.stats


def accuracy(correct, total) -> numpy.ndarray:
    return numpy.around(correct / total * 100, decimals=2)


def best_val_acc(accuracy_traces: List[List[float]]):
    return _potentially_pad(accuracy_traces).max(axis=1, initial=0).mean()


def best_test_acc(accuracies):
    return numpy.array(accuracies).mean()


def conf_h_test_acc(accuracies, alpha=0.05):
    a = numpy.array(accuracies).astype(numpy.float64)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((2 - alpha) / 2., len(a) - 1)

    return h


def best_val_acc_epoch(accuracy_traces: List[List[float]]):
    return _potentially_pad(accuracy_traces).argmax(axis=1).mean()


def best_loss(loss_traces: List[List[float]]):
    return _potentially_pad(loss_traces).min(axis=1, initial=0).mean()


def _potentially_pad(array, pad_val=0) -> numpy.ndarray:
    max_len = max([len(l) for l in array])
    if not all(len(l) == len(array[0]) for l in iter(array)):
        array = numpy.array([numpy.pad(x, pad_width=(0, max_len - len(x)), constant_values=pad_val) for x in array])
    else:
        array = numpy.array(array)

    return array


def confidence_around_mean(data, alpha=0.05):
    a = numpy.array(data).astype(numpy.float64)
    m, se = numpy.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((2 - alpha) / 2., len(a) - 1)
    return m, m - h, m + h
