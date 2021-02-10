from typing import Tuple

import numpy
import torch
from torch import nn
from torch.nn.modules.loss import _Loss


def evaluate_classification(model: nn.Module,
                            data_loader,
                            criterion: _Loss = None,
                            device: torch.device = "cpu") -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Evaluate a model on a dataset (in a data loader). Return (correct, total)."""
    if hasattr(data_loader.dataset, "classes"):
        correct = torch.zeros(len(data_loader.dataset.classes))
        total = torch.zeros(len(data_loader.dataset.classes))
    elif hasattr(data_loader.dataset.dataset, "classes"):
        correct = torch.zeros(len(data_loader.dataset.dataset.classes))
        total = torch.zeros(len(data_loader.dataset.dataset.classes))
    elif hasattr(data_loader.dataset, "labels"):
        correct = torch.zeros(numpy.unique(data_loader.dataset.labels).size)
        total = torch.zeros(numpy.unique(data_loader.dataset.labels).size)
    elif hasattr(data_loader.dataset.dataset, "labels"):
        correct = torch.zeros(numpy.unique(data_loader.dataset.dataset.labels).size)
        total = torch.zeros(numpy.unique(data_loader.dataset.dataset.labels).size)
    else:
        raise AttributeError("Given DataLoader contains unexpected dataset "
                             "structure preventing reading class labels.")

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
