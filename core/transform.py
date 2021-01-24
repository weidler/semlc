from typing import Tuple, Union, List

from torchvision import transforms as transforms

NORMALIZATION_MEANS, NORMALIZATION_STDS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def make_transform_composition(size: Union[int, Tuple[int, int]], channels: int, augmentations: List[object] = None):
    sizes = (size, size) if isinstance(size, int) else size

    transformations = [transforms.RandomCrop(sizes, 4),
                       transforms.RandomHorizontalFlip()]
    transformations += augmentations if augmentations is not None else []

    transformations.append(transforms.ToTensor())

    # finish with normalization
    transformations.append(transforms.Normalize(NORMALIZATION_MEANS[:channels], NORMALIZATION_STDS[:channels]))

    return transforms.Compose(transformations)


def make_test_transform_composition(size: Union[int, Tuple[int, int]], channels: int, augmentations: List[object] = None):
    sizes = (size, size) if isinstance(size, int) else size

    transformations = [transforms.CenterCrop(sizes)]
    transformations += augmentations if augmentations is not None else []

    transformations.append(transforms.ToTensor())

    # finish with normalization
    transformations.append(transforms.Normalize(NORMALIZATION_MEANS[:channels], NORMALIZATION_STDS[:channels]))

    return transforms.Compose(transformations)
