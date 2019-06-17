import time

import torch


def div_complex_iterative(a: torch.Tensor, b: torch.Tensor):
    """Iterative division of complex tensors. Implemented to see if my approach works."""
    assert a.shape[0] == b.shape[0]
    assert a.shape[1] == 2 and b.shape[1] == 2

    result = torch.zeros(a.shape)
    for i in range(a.shape[0]):
        numerator = a[i]
        denominator = b[i]

        # multiply by complex conjugate to make denominator real
        denominator_real = denominator[0]**2 + (-denominator[1])**2
        numerator_real = (numerator[0] * denominator[0]) - (numerator[1] * (-denominator[1]))
        numerator_complex = (numerator[0] * (-denominator[1])) + (numerator[1] * denominator[0])

        result[i][0] = numerator_real/denominator_real
        result[i][1] = numerator_complex/denominator_real

    return result


def div_complex_vectorized(numerator: torch.Tensor, denominator: torch.Tensor):
    """Vectorized division of complex tensors."""
    assert numerator.shape[0] == denominator.shape[0]
    assert numerator.shape[1] == 2 and denominator.shape[1] == 2

    result = torch.zeros(numerator.shape)
    denominator_real = denominator[:, 0] ** 2 + (-denominator[:, 1]) ** 2
    numerator_real = (numerator[:, 0] * denominator[:, 0]) - (numerator[:, 1] * (-denominator[:, 1]))
    numerator_complex = (numerator[:, 0] * (-denominator[:, 1])) + (numerator[:, 1] * denominator[:, 0])

    result[:, 0] = numerator_real / denominator_real
    result[:, 1] = numerator_complex / denominator_real

    return result


def div_complex(numerator: torch.Tensor, denominator: torch.Tensor):
    """Optimized vectorized complex tensor division.

    Shape of denominator and numerator must match. Input shape: N x C x H x W."""
    # assert numerator.shape[-2] == denominator.shape[-2]
    assert numerator.shape[-1] == 2 and denominator.shape[-1] == 2

    denominator_real = denominator[:, :, :, :, 0] ** 2 + (-denominator[:, :, :, :, 1]) ** 2
    numerator_real = (numerator[:, :, :, :, 0] * denominator[:, :, :, :, 0]) - (numerator[:, :, :, :, 1] * (-denominator[:, :, :, :, 1]))
    numerator_complex = (numerator[:, :, :, :, 0] * (-denominator[:, :, :, :, 1])) + (numerator[:, :, :, :, 1] * denominator[:, :, :, :, 0])

    return torch.stack((numerator_real/denominator_real, numerator_complex/denominator_real), dim=-1)


if __name__ == "__main__":
    # sanity check
    t = torch.randn((14, 14, 100, 2))

    # benchmarking
    a = torch.randn((1000, 2))
    b = torch.randn((1000, 2))

    for method in [div_complex_iterative, div_complex_vectorized, div_complex]:
        start = time.time()
        for i in range(100):
            method(a, b)
        print(f"{method.__name__}: {round(time.time() - start, 2)}s")
