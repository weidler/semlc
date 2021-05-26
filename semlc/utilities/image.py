from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from utilities.util import closest_factors


def show_image(img, mean=0.5, std=0.5):
    img = img * mean + std
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_image_gray(I, block=True, **kwargs):
    # utility function to show image
    plt.figure()
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)
    plt.show(block=block)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def grayify_rgb_filters(rgb_filters: np.ndarray) -> np.ndarray:
    """Approximate grayscale filters from RGB filters.

    Input Shape:  [N, C, W, H]
    """

    return np.dot(np.swapaxes(rgb_filters[:, :3, ...], 1, -1), [0.2989, 0.5870, 0.1140])


def grid_plot(imgs: list, name=None, block=True, range=None):
    rs, cs = closest_factors(len(imgs))
    fig, axs = plt.subplots(
        nrows=rs,
        ncols=cs,
    )

    fig.set_size_inches(cs * 3, rs * 4)

    i = 0
    for row in axs:
        for col in row:
            col.imshow(imgs[i].squeeze(), cmap=plt.gray(),
                       **(dict(vmin=range[0], vmax=range[1]) if range is not None else dict()), interpolation="none")
            col.axis("off")

            i += 1

            if len(imgs) - 1 < i:
                break
        if len(imgs) - 1 < i:
            break

    if name is not None:
        fig.suptitle(name)


def row_plot(imgs: list, name=None, block=True, range=None, labels=None, cmap=plt.gray()):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(imgs),
    )

    fig.set_size_inches(4 * len(imgs), 4)

    i = 0
    for col in axs:
        col.imshow(imgs[i], cmap=cmap, **(dict(vmin=range[0], vmax=range[1]) if range is not None else dict()))
        col.axis("off")
        if labels is not None:
            col.set_title(labels[i], fontsize=16)
        i += 1

    if name is not None:
        plt.suptitle(name, weight="bold")
