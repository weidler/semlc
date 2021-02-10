import numpy
import torch

from numpy.core.multiarray import ndarray


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def seq_to_img(seq: ndarray):
    """ Convert a CIFAR image sequence into a ndarray of shape (row, col, channel).

    :param seq:     the original sequence of 3072 values, given as a row in CIFAR.
    """
    img = ndarray((32, 32, 3))
    for row in range(32):
        for col in range(32):
            for channel in range(3):
                img[row, col, channel] = seq[(channel * 1024) + (row * 32) + col]

    return img.astype(int)


def seq_to_tensor(seq: ndarray):
    """ Convert a CIFAR image sequence into a pytorch tensor of shape (row, col, channel).

    :param seq:     the original sequence of 3072 values, given as a row in CIFAR.
    """
    tensor = torch.tensor(seq).float()
    tensor = tensor.view((3, -1))
    tensor = tensor.view((3, 32, 32))
    tensor = tensor.unsqueeze(0)/255

    return tensor


def tensor_to_img(tensor: torch.Tensor):
    tensor.squeeze_()
    img = ndarray((32, 32, 3))
    for row in range(32):
        for col in range(32):
            for channel in range(3):
                img[row, col, channel] = tensor[channel, row, col] * 255

    return img.astype(int)


def img_to_tensor(image):
    image = numpy.array(image)
    tensor = torch.ones((1, 3, 32, 32)).float()
    for row in range(32):
        for col in range(32):
            for channel in range(3):
                tensor[0, channel, row, col] = image[row, col, channel] / 255

    return tensor
