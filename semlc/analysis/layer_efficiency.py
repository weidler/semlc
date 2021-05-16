"""Experiment script that compares the inhibition groups regarding their computational efficiency."""
import functools
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import gaussian
from torch import nn, optim
from torch.nn.functional import mse_loss
from tqdm import tqdm

from layers import AdaptiveSemLC, SemLC, \
    SingleShotSemLC, ParametricSemLC
from layers.semantic_layers_fft import FFTAdaptiveSemLC, FFTSemLC


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

    fw_time = 0
    bw_time = 0
    loss_time = 0

    target = torch.randn(layer(tensor_in).shape)
    total_start_time = time.time()
    for i in range(n):
        if has_parameters:
            start_time = time.time()
            optimizer.zero_grad()
            bw_time += time.time() - start_time

        start_time = time.time()
        out = layer(tensor_in)
        fw_time += time.time() - start_time

        start_time = time.time()
        loss = mse_loss(out, target)
        loss_time += time.time() - start_time

        if has_parameters:
            start_time = time.time()
            loss.backward()
            optimizer.step()
            bw_time += time.time() - start_time

    return time.time() - total_start_time, fw_time, bw_time, loss_time


def make_layer_functions():
    return [
        "conv2d",
        functools.partial(SingleShotSemLC, ricker_width=wavelet_width, ricker_damp=damping),
        functools.partial(SemLC, ricker_width=wavelet_width, ricker_damp=damping),
        functools.partial(AdaptiveSemLC, ricker_width=wavelet_width, ricker_damp=damping),
        functools.partial(ParametricSemLC, ricker_width=wavelet_width, ricker_damp=damping),
        functools.partial(FFTAdaptiveSemLC, ricker_width=wavelet_width, ricker_damp=damping),
        functools.partial(FFTSemLC, ricker_width=wavelet_width, ricker_damp=damping),
    ]


def make_network(layer, depth):
    conv_one = nn.Conv2d(3, depth, kernel_size=5, stride=1, padding=2)
    tested_layer = layer(hooked_conv=conv_one) if not layer == "conv2d" else nn.Sequential(nn.ReLU(inplace=True),
                                                                                           nn.Conv2d(depth, depth, 3, 1))

    return nn.Sequential(
        conv_one,
        tested_layer,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(depth, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=3, stride=2)
    )


def make_input():
    tensor_in = torch.zeros((128, 3, 24, 24))
    for b in range(128):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(gaussian(3, 6))

    return tensor_in


# SETTINGS
n_passes = 100
depth_x = [16, 32, 64, 128]
width = 28
height = 28
wavelet_width = 6
damping = 0.12

tensor_in = make_input()

# RANKED LAYERS
print("Calculating Layer Ranking")
results_adaptive_scope = {}
results_constant_scope = {}

depth = depth_x[2]
scope = depth - 1

results = []
for test_layer in tqdm(make_layer_functions(), desc="Ranking"):
    net = make_network(test_layer, depth)

    execution_time, fw_time, bw_time, loss_time = make_passes(net, n_passes)
    results.append((test_layer.func.__name__ if not isinstance(test_layer, str) else test_layer,
                    round(execution_time / n_passes, 4),
                    # round(fw_time / n_passes, 4),
                    # round(bw_time / n_passes, 4),
                    # round(loss_time / n_passes, 4),
                    ))

# ranking
ranked_performance = sorted(results, key=lambda x: x[1])
for i, (name, t) in enumerate(ranked_performance, 1):
    print(f"{i}.\t{name} with {t}s.")

# save latex table
# df = DataFrame(ranked_performance)
# df.index += 1
# with open("./documentation/tables/efficiency.tex", "w") as f:
#     f.write(df.to_latex(header=["Strategy", "Time (s)"]))

exit()

# ADAPTIVE DEPTH
print("Calculating Adaptive Depth Times")
for depth in tqdm(depth_x, desc="Adaptive Depth Benchmark"):
    scope = depth - 1

    for test_layer in make_layer_functions():
        net = make_network(test_layer, depth)
        execution_time = make_passes(net, n_passes)

        layer_name = test_layer.__class__.__name__
        if layer_name not in results_adaptive_scope:
            results_adaptive_scope.update({layer_name: []})
        results_adaptive_scope[layer_name].append(execution_time)

# FIXED DEPTH
print("\nCalculating Fixed Depth Times")
for depth in tqdm(depth_x, desc="Constant Depth Benchmark"):
    scope = depth_x[0] - 1

    for test_layer in make_layer_functions():
        net = make_network(test_layer, depth)
        execution_time = make_passes(net, n_passes)

        layer_name = test_layer.__class__.__name__
        if layer_name not in results_constant_scope:
            results_constant_scope.update({layer_name: []})
        results_constant_scope[layer_name].append(execution_time)

# PLOT RESULTS
axs: List[Axes]
fig: Figure
fig, axs = plt.subplots(1, 2)
fig.set_size_inches(10, 7)
for layer_name in results_adaptive_scope.keys():
    axs[0].plot(depth_x, results_adaptive_scope[layer_name], label=name_map[layer_name])
    axs[1].plot(depth_x, results_constant_scope[layer_name], label=name_map[layer_name])

subtitle_font = {"style": "oblique", "size": 11}
axs[0].set_title("Adaptive Scope", subtitle_font)
axs[1].set_title("Constant Scope", subtitle_font)
axs[0].set_ylabel("Execution Time")
axs[1].set_ylabel("Execution Time")
axs[0].set_xlabel("Number of Filters")
axs[1].set_xlabel("Number of Filters")

axs[0].set_xticks(depth_x)
axs[1].set_xticks(depth_x)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.subplots_adjust(hspace=0.35, bottom=0.2)

plt.savefig('./layer_efficiency_horizontal.pdf', format="pdf", bbox_inches='tight')
plt.show()
