from typing import Tuple

import numpy
import torchvision
from config import CONFIG
from core.transform import make_transform_composition

AVAILABLE_DATASETS = ["cifar10", "mnist", "fashionmnist", "svhn", "kylberg", "rectangles"]


def get_dataset_class(name: str):
    assert name.lower() in AVAILABLE_DATASETS

    return {
        "cifar10": torchvision.datasets.CIFAR10,
        "mnist": torchvision.datasets.MNIST,
        "fashionmnist": torchvision.datasets.FashionMNIST,
    }[name.lower()]


def get_training_dataset(name: str, force_crop: Tuple[int, int] = None):
    name = name.lower()

    if name == "cifar10":
        width, height = (32, 32)
        dataset = torchvision.datasets.CIFAR10(root=CONFIG.DATA_DIR, train=True, download=True,
                                               transform=make_transform_composition(
                                                   (width, height) if force_crop is None else force_crop, 3))
    elif name == "mnist":
        width, height = (28, 28)
        dataset = torchvision.datasets.MNIST(root=CONFIG.DATA_DIR, train=True, download=True,
                                             transform=make_transform_composition(
                                                 (width, height) if force_crop is None else force_crop, 1))
    elif name == "fashionmnist":
        width, height = (28, 28)
        dataset = torchvision.datasets.FashionMNIST(root=CONFIG.DATA_DIR, train=True, download=True,
                                                    transform=make_transform_composition(
                                                        (width, height) if force_crop is None else force_crop, 1))
    else:
        raise NotImplementedError("Unknown dataset.")

    return dataset


def get_class_labels(dataset) -> list:
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "labels"):
        return numpy.unique(dataset.labels).tolist()
    elif hasattr(dataset, "dataset"):
        return get_class_labels(dataset.dataset)
    else:
        raise ValueError("Cannot handle given dataset.")


def get_number_of_classes(dataset) -> int:
    return len(get_class_labels(dataset))


if __name__ == '__main__':
    cifar10 = get_training_dataset("cifar10")
    mnist = get_training_dataset("mnist")
    fashionmnist = get_training_dataset("fashionmnist")
    svhn = get_training_dataset("svhn")
    kylberg = get_training_dataset("kylberg")