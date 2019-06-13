import time

import torch


def toeplitz1d_iterative(k: torch.Tensor, m: int):
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


def toeplitz1D(k: torch.Tensor, m: int):
    assert len(k.shape) == 1

    n = len(k)
    ttype = k.dtype
    k = k.view(1, -1)

    rows = []
    for i in range(m):
        rows.append(
            torch.cat((torch.zeros((1, i)).type(dtype=ttype), k, torch.zeros((1, m - i - 1)).type(dtype=ttype)),
                      dim=1))

    return torch.cat((rows), dim=0)


if __name__ == "__main__":
    kernel = torch.tensor([1, 2, 3, 4, 5])

    for method in [toeplitz1D, toeplitz1d_iterative]:
        start = time.time()
        for i in range(10000):
            method(kernel, 5)
        print(f"{method.__name__}: {round(time.time() - start, 2)}s")

    print(toeplitz1D(kernel, 3))
