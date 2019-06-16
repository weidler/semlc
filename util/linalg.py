import time

import torch


def toeplitz1d_iterative(k: torch.Tensor, m: int):
    """Deprecated."""
    assert len(k.shape) == 1

    n = len(k)
    ttype = k.dtype
    k = k.view(1, -1)

    tpl_matrix = torch.cat((k, torch.zeros((1, m - 1)).type(dtype=ttype)), dim=1)
    for i in range(1, m):
        new_row = torch.cat(
            (torch.zeros((1, i)).type(dtype=ttype), k, torch.zeros((1, m - i - 1)).type(dtype=ttype)), dim=1)
        tpl_matrix = torch.cat((tpl_matrix, new_row), dim=0)

    return tpl_matrix


def toeplitz1d(k: torch.Tensor, m: int) -> torch.Tensor:
    """Toeplitz matrix of given filter k for 1D convolution. Convolution is centered at leftmost index of k.

    For a convolution of signal s with filter k,
        s * k,
    the equivalent operation with Toeplitz matrix K is
        s.K
    where a.B is the matrix multiplication between a and B.

    :param k:       one-dimensional filter k, given as tensor, left-centered
    :param m:       the size of the signal s, specifying the height of the Toeplitz matrix
    :return:        the toeplitz matrix as a Tensor
    """
    assert len(k.shape) == 1

    n = len(k)
    k = k.view(1, -1)
    rows = []
    for i in range(m):
        rows.append(k.roll(i, dims=1))

    return torch.cat(rows, dim=0).t()


if __name__ == "__main__":
    kernel = torch.tensor([1, 2, 3, 4, 5])

    for method in [toeplitz1d, toeplitz1d_iterative]:
        start = time.time()
        for i in range(10000):
            method(kernel, 5)
        print(f"{method.__name__}: {round(time.time() - start, 2)}s")

    print(toeplitz1d(kernel, 3))
