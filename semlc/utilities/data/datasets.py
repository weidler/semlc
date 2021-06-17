import glob
import socket
from typing import Tuple, Union

import numpy
import torch.utils.data
import torchvision
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from config import CONFIG
from core.transform import make_transform_composition, make_test_transform_composition
from utilities.data.imagenet import imagenet_dali_dataloader, DALITorchLoader

AVAILABLE_DATASETS = ["cifar10", "cifar10-bw", "mnist", "imagenet"]


def get_dataset_class(name: str):
    assert name.lower() in AVAILABLE_DATASETS

    return {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar10-bw": torchvision.datasets.CIFAR10,
        "mnist": torchvision.datasets.MNIST,
    }[name.lower()]


def get_training_dataset(name: str,
                         force_size: Tuple[int, int] = None) -> Tuple[Union[DataLoader, DALITorchLoader],
                                                                      Union[DataLoader, DALITorchLoader]]:
    name = name.lower()

    train_set_loader, validation_set_loader = None, None
    extract_loaders = True
    if name == "cifar10":
        width, height = (32, 32)
        dataset = torchvision.datasets.CIFAR10(root=CONFIG.DATA_DIR, train=True, download=True,
                                               transform=make_transform_composition(
                                                   (width, height) if force_size is None else force_size, 3))
    elif name == "cifar10-bw":
        width, height = (28, 28)
        dataset = torchvision.datasets.CIFAR10(root=CONFIG.DATA_DIR, train=True, download=True,
                                               transform=make_transform_composition(
                                                   size=(width, height) if force_size is None else force_size,
                                                   channels=1,
                                                   augmentations=[transforms.Grayscale()]))
    elif name == "mnist":
        width, height = (28, 28)
        transform = transforms.Compose([
            transforms.Pad(2),
            transforms.RandomCrop((width, height)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = torchvision.datasets.MNIST(root=CONFIG.DATA_DIR, train=True, download=True, transform=transform)
    else:
        extract_loaders = False

    if extract_loaders:
        train_set, validation_set = random_split(dataset,
                                                 [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])

        train_set_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, )
        validation_set_loader = DataLoader(validation_set, batch_size=128, shuffle=True, num_workers=2)

    if name == "imagenet":
        if "daint" in socket.gethostname():
            train_recs = '/scratch/snx3000/datasets/imagenet/ILSVRC2012_1k/train/*'
            train_idx = '/scratch/snx3000/datasets/imagenet/ILSVRC2012_1k/idx_files/train/*'
            val_recs = '/scratch/snx3000/datasets/imagenet/ILSVRC2012_1k/validation/*'
            val_idx = '/scratch/snx3000/datasets/imagenet/ILSVRC2012_1k/idx_files/validation/*'
        else:
            train_recs = '../data/imagenet/train/*'
            train_idx = '../data/imagenet/idx_files/train/*'
            val_recs = '../data/imagenet/validation/*'
            val_idx = '../data/imagenet/idx_files/validation/*'

        train_set_loader = imagenet_dali_dataloader(
            sorted(glob.glob(train_recs)),
            sorted(glob.glob(train_idx)),
            batch_size=128,
            training=True)

        validation_set_loader = imagenet_dali_dataloader(
            sorted(glob.glob(val_recs)),
            sorted(glob.glob(val_idx)),
            batch_size=128,
            training=True)

    if train_set_loader is None:
        raise ValueError("Unknown Dataset.")

    return train_set_loader, validation_set_loader


def load_test_set(image_channels: int, image_height: int, image_width: int, dataset: str):
    """Return all available test dataset variants for given dataset."""
    dataset_class = get_dataset_class(dataset)

    if dataset.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset.lower() == "cifar10":
        transform = make_test_transform_composition((image_width, image_height), image_channels)
    elif dataset.lower() == "cifar10-bw":
        transform = make_test_transform_composition((image_width, image_height), image_channels,
                                                    augmentations=[transforms.Grayscale()])
    else:
        raise ValueError("Unknown Dataset.")

    return {"default": dataset_class(root=CONFIG.DATA_DIR, train=False, download=True,
                                     transform=transform)}


def get_class_labels(dataset) -> list:
    if hasattr(dataset, "classes"):
        return dataset.classes
    elif hasattr(dataset, "labels"):
        return numpy.unique(dataset.labels).tolist()
    elif hasattr(dataset, "dataset"):
        return get_class_labels(dataset.dataset)
    else:
        raise ValueError("Cannot handle given dataset.")


def get_number_of_classes(dataloader: Union[DataLoader, DALITorchLoader]) -> int:
    if isinstance(dataloader, DataLoader):
        return len(get_class_labels(dataloader.dataset))
    else:
        return dataloader.n_classes


if __name__ == '__main__':
    cifar10 = get_training_dataset("cifar10")
    mnist = get_training_dataset("mnist")
