import math

import torch

from util.complex import div_complex


def pad_roll(k: torch.Tensor, in_channels, scope):
    """Zero-pad around filter, then roll to have center at i=0. Need to use concatenation to keep padding out of
    auto grad functionality. If torch's pad() function would be used, padding can be adjusted during optimization."""
    pad_left = torch.zeros((1, 1, (in_channels - scope) // 2))
    pad_right = torch.zeros((1, 1, (in_channels - scope) - pad_left.shape[-1]))

    return torch.cat((pad_left, k, pad_right), dim=-1).roll(math.floor(in_channels / 2) + 1)


def toeplitz1d_circular(k: torch.Tensor, m: int) -> torch.Tensor:
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
    assert len(k.shape) == 1, f"Filter needs to be a vector to be converted to a Toeplitz matrix for 1D convolution. " \
                              f"Given filter k is a tensor with {len(k.shape)} domains."

    k = pad_roll(k.view(1, 1, -1), in_channels=m, scope=k.shape[0]).view(1, -1)
    rows = []
    for i in range(m):
        rows.append(k.roll(i, dims=1))

    return torch.cat(rows, dim=0)


def toeplitz1d_zero(k: torch.Tensor, m: int) -> torch.Tensor:
    """Toeplitz matrix of given filter k for 1D convolution. Convolution is centered at center index.

    For a convolution of signal s with filter k,
        s * k,
    the equivalent operation with Toeplitz matrix K is
        s.K
    where a.B is the matrix multiplication between a and B.

    :param k:       raw one-dimensional filter k, given as tensor
    :param m:       the size of the signal s, specifying the height of the Toeplitz matrix
    :return:        the toeplitz matrix as a Tensor
    """
    assert len(k.shape) == 1, f"Filter needs to be a vector to be converted to a Toeplitz matrix for 1D convolution. " \
                              f"Given filter k is a tensor with {len(k.shape)} domains."
    assert k.shape[0] % 2 != 0, f"Given filter is of size {k.shape[0]}, but centering requires the filter to have an " \
                                f"odd number of dimensions."

    n = k.shape[0]
    k = k.view(1, -1)
    rows = []
    source_row = torch.cat((torch.zeros((1, m), dtype=k.dtype), k, torch.zeros((1, m), dtype=k.dtype)), dim=1)
    start_index = (n // 2) + m
    for i in range(m):
        rows.append(source_row[:, start_index - i: start_index - i + m])

    return torch.cat(rows, dim=0)


def convolve_3d_toeplitz(tpl_matrix: torch.Tensor, signal_tensor: torch.Tensor):
    # stack activation depth-columns for depth-wise convolution
    stacked_activations = signal_tensor.unbind(dim=2)
    stacked_activations = torch.cat(stacked_activations, dim=2).permute((0, 2, 1))

    # convolve by multiplying with tpl
    convolved_tensor = stacked_activations.matmul(tpl_matrix)
    convolved_tensor = convolved_tensor.permute((0, 2, 1))

    # recover original shape
    return convolved_tensor.view_as(signal_tensor)


def convolve_3d_fourier(filter: torch.Tensor, signal: torch.Tensor, delta: torch.Tensor):
    # bring the dimension that needs to be fourier transformed to the end
    signal = signal.permute((0, 2, 3, 1))

    # fourier transform
    fourier_activations = torch.rfft(signal, 1, onesided=False)
    fourier_filter = torch.rfft(delta - filter, 1, onesided=False)

    # divide in frequency domain, then bring back to time domain
    convolved_tensor = torch.irfft(div_complex(fourier_activations, fourier_filter), 1, onesided=False)

    # restore original shape
    convolved_tensor = convolved_tensor.permute((0, 3, 1, 2))

    return convolved_tensor
