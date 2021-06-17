from typing import Tuple, Union

import numpy
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from utilities.data.datasets import get_number_of_classes
from utilities.data.imagenet import DALITorchLoader


def evaluate_classification(model: nn.Module,
                            data_loader: Union[DataLoader, DALITorchLoader],
                            criterion: _Loss = None,
                            device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Evaluate a model on a dataset (in a data loader). Return (correct, total)."""
    n_classes = get_number_of_classes(data_loader)

    correct = torch.zeros(n_classes)
    total = torch.zeros(n_classes)

    with torch.no_grad():
        batch = 0
        total_loss = torch.tensor(0.0)
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            total_loss = total_loss + criterion(outputs, labels) if criterion is not None else 0
            _, predicted = torch.max(outputs.data, 1)

            for c in range(len(correct)):
                is_correct = predicted[torch.where(labels == c)] == c
                total[c] = total[c] + len(is_correct)
                correct[c] = correct[c] + is_correct.sum()

            batch += 1

    if criterion is None:
        return correct, total, numpy.inf
    else:
        return correct, total, (total_loss / batch).item()
