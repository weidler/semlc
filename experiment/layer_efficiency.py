"""Experiment script that compares the inhibition strategies regarding their computational efficiency."""

import random
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.signal import gaussian
from torch import nn, optim
from torch.nn.functional import mse_loss
from tqdm import tqdm

from model.inhibition_layer import SingleShotInhibition, RecurrentInhibition, ConvergedInhibition, \
    ConvergedToeplitzInhibition, ConvergedFrozenInhibition, ConvergedToeplitzFrozenInhibition


def make_passes(layer, n):
    optimizer = None
    has_parameters = len(list(test_layer.parameters())) > 0
    if has_parameters:
        optimizer = optim.SGD(test_layer.parameters(), 0.01)
    start_time = time.time()

    for i in range(n):
        if has_parameters:
            optimizer.zero_grad()

        out = layer(tensor_in)
        target = out * random.random()

        loss = mse_loss(out, target)

        if has_parameters:
            loss.backward()
            optimizer.step()
    return round(time.time() - start_time, 2)


# SETTINGS
batches = 1
n_forward_passes = 100
depth_x = [16, 32, 64, 128, 256]
width = 14
height = 14
wavelet_width = 6
damping = 0.12

# BENCHMARK
results_adaptive_scope = {}
results_constant_scope = {}

depth = depth_x[3]
scope = depth - 1

tensor_in = torch.zeros((batches, depth, width, height))
for b in range(batches):
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[b, :, i, j] = torch.from_numpy(gaussian(depth, 6))

simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)
inhibitor = SingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
inhibitor_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
inhibitor_conv = ConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_conv_freeze = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
inhibitor_tpl_freeze = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)

results = []
for test_layer in tqdm([simple_conv, inhibitor, inhibitor_rec, inhibitor_conv, inhibitor_conv_freeze, inhibitor_tpl,
                        inhibitor_tpl_freeze]):
    execution_time = make_passes(test_layer, n_forward_passes)
    results.append((test_layer.__class__.__name__, execution_time))

ranked_performance = sorted(results, key=lambda x: x[1])
for i, (name, t) in enumerate(ranked_performance, 1):
    print(f"{i}.\t{name} with {t}s.")

df = DataFrame(ranked_performance)
df.index += 1
with open("../documentation/tables/efficiency.tex", "w") as f:
    f.write(df.to_latex(header=["Strategy", "Time (s)"]))


# Adaptive depth
for depth in tqdm(depth_x, desc="Adaptive Depth Benchmark"):
    scope = depth - 1

    tensor_in = torch.zeros((1, depth, width, height))
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[0, :, i, j] = torch.from_numpy(gaussian(depth, 4))

    simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)
    inhibitor = SingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_conv = ConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_conv_freeze = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl_freeze = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    for test_layer in [simple_conv, inhibitor, inhibitor_rec, inhibitor_conv, inhibitor_tpl, inhibitor_conv_freeze, inhibitor_tpl_freeze]:
        execution_time = make_passes(test_layer, n_forward_passes)
        layer_name = test_layer.__class__.__name__
        if layer_name not in results_adaptive_scope:
            results_adaptive_scope.update({layer_name: []})
        results_adaptive_scope[layer_name].append(execution_time)

# Fixed depth
for depth in tqdm(depth_x, desc="Constant Depth Benchmark"):
    scope = depth_x[0] - 1

    tensor_in = torch.zeros((1, depth, width, height))
    for i in range(tensor_in.shape[-1]):
        for j in range(tensor_in.shape[-2]):
            tensor_in[0, :, i, j] = torch.from_numpy(gaussian(depth, 4))

    simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)
    inhibitor = SingleShotInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_rec = RecurrentInhibition(scope, wavelet_width, damp=damping, padding="zeros", learn_weights=True)
    inhibitor_conv = ConvergedInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl = ConvergedToeplitzInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_conv_freeze = ConvergedFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    inhibitor_tpl_freeze = ConvergedToeplitzFrozenInhibition(scope, wavelet_width, damp=damping, in_channels=depth)
    for test_layer in [simple_conv, inhibitor, inhibitor_rec, inhibitor_conv, inhibitor_tpl, inhibitor_conv_freeze, inhibitor_tpl_freeze]:
        execution_time = make_passes(test_layer, n_forward_passes)
        layer_name = test_layer.__class__.__name__
        if layer_name not in results_constant_scope:
            results_constant_scope.update({layer_name: []})
        results_constant_scope[layer_name].append(execution_time)


# PLOT RESULTS
axs: List[Axes]
fig: Figure
fig, axs = plt.subplots(2, 1)
fig.set_size_inches(5, 9)
for layer_name in results_adaptive_scope.keys():
    axs[0].plot(depth_x, results_adaptive_scope[layer_name], label=layer_name)
    axs[1].plot(depth_x, results_constant_scope[layer_name], label=layer_name)

subtitle_font = {"style": "oblique", "size": 11}
axs[0].set_title("Adaptive Scope", subtitle_font)
axs[1].set_title("Constant Scope", subtitle_font)
axs[0].set_ylabel("Execution Time")
axs[1].set_ylabel("Execution Time")
axs[0].set_xlabel("Number of Filters")
axs[1].set_xlabel("Number of Filters")

depth_x.remove(32)
axs[0].set_xticks(depth_x)
axs[1].set_xticks(depth_x)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.subplots_adjust(hspace=0.35, bottom=0.2)

plt.savefig('../documentation/figures/layer_efficiency.pdf', format="pdf", bbox_inches='tight')
plt.show()
