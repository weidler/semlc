"""Script checks if the layers behave as intended (regarding learning)."""

import time

import torch
from scipy.signal import gaussian
from torch import nn, optim
from torch.nn.functional import mse_loss

from model.inhibition_layer import ConvergedInhibition, ConvergedFrozenInhibition
from model.fft_inhibition_layer import FFTConvergedInhibition, FFTConvergedFrozenInhibition
from model.deprecated_inhibition_layer import Conv3DSingleShotInhibition, Conv3DRecurrentInhibition

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
print(f"USE CUDA: {use_cuda}.")


def make_passes(layer, n):
    optimizer = None
    has_parameters = len(list(layer.parameters())) > 0
    if has_parameters:
        optimizer = optim.SGD(layer.parameters(), 0.01)
    start_time = time.time()

    for i in range(n):
        if has_parameters:
            optimizer.zero_grad()

        out = layer(tensor_in)
        target = torch.randn(out.shape)

        loss = mse_loss(out, target)

        if has_parameters:
            loss.backward()
            optimizer.step()

    return round(time.time() - start_time, 2)


# SETTINGS
batches = 128
n_forward_passes = 1
width = 28
height = 28
wavelet_width = 6
damping = 0.12

# BENCHMARK
results_adaptive_scope = {}
results_constant_scope = {}

depth = 64
scope = depth - 1

tensor_in = torch.zeros((batches, 3, width, height))
for b in range(batches):
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[b, :, i, j] = torch.from_numpy(gaussian(3, 6))

simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)
inhibitor = Conv3DSingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
inhibitor_rec = Conv3DRecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True,
                                          max_steps=5)
inhibitor_conv = FFTConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_conv_freeze = FFTConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_tpl = ConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_tpl_freeze = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)

for test_layer in [simple_conv, inhibitor, inhibitor_rec, inhibitor_conv, inhibitor_conv_freeze, inhibitor_tpl,
                   inhibitor_tpl_freeze]:
    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # 0
        test_layer,  # 1
        nn.ReLU(inplace=True),  # 2
        nn.MaxPool2d(kernel_size=3, stride=2),  # 3

        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 4
        nn.ReLU(inplace=True),  # 5
        nn.AvgPool2d(kernel_size=3, stride=2)  # 6
    )

    names = [n[0] for n in list(net.named_parameters())]
    before = [p.clone() for p in list(net.parameters())]
    execution_time = make_passes(net, n_forward_passes)
    after = [p.clone() for p in list(net.parameters())]

    print(f"\nStayed the same for {test_layer.__class__.__name__}:")
    for (n, b, a) in zip(names, before, after):
        print(f"{n}: {bool(torch.all(a == b).item())}")
