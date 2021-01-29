import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from layers.semantic_layers import SingleShotSemLC, AdaptiveSemLC, SemLC, ParametricSemLC, GaussianSemLC, LRN, CMapLRN, \
    SemLCLRNChain, LRNSemLCChain


def lateral_pass_plot(tensor_out, label, line_style="."):
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), line_style, label=label)


if __name__ == "__main__":
    from scipy.signal import gaussian

    # CUDA
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"USE CUDA: {use_cuda}.")

    # SETTINGS
    batches = 1
    depth = 64
    scope = depth - 1
    width = 14
    height = 14
    wavelet_width = 3
    damping = 0.2
    self_connect = True
    match_lrn_peak = False

    tensor_in = torch.zeros((batches, depth, width, height))
    for b in range(batches):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(
                    gaussian(depth, wavelet_width)
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), -(scope // 4)) * 0.2
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), (scope // 4)) * 0.2
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), -(scope // 2)) * 0.1
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), (scope // 2)) * 0.1
                )

    tensor_in *= 100

    simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)

    layers = [
        # SingleShotSemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping, learn_weights=True, self_connection=self_connect),

        # circular padding
        SemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping * 100, self_connection=self_connect),
        SemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping * 2, self_connection=self_connect),
        SemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping, self_connection=self_connect),
        SemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping * 0.5, self_connection=self_connect),
        SemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping * 0.5 * 0.5, self_connection=self_connect),
        # GaussianSemLC(simple_conv, ricker_width=wavelet_width, ricker_damp=damping, self_connection=self_connect),
        # LRN(simple_conv, ricker_width=wavelet_width, ricker_damp=damping),

        # SemLCLRNChain(simple_conv, ricker_width=wavelet_width, ricker_damp=damping),
        # LRNSemLCChain(simple_conv, ricker_width=wavelet_width, ricker_damp=damping),
    ]

    line_styles = ["-", "--", ".-", "-"]
    line_styles += ["-" for i in range(len(layers) - len(line_styles))]

    tensor_outs = {l.name: l(tensor_in) for l in layers}

    plt.clf()
    input_factor_lrn = (tensor_outs["LRN"][:, 31] / tensor_in[:, 31]) if match_lrn_peak else 1
    plt.plot((tensor_in * input_factor_lrn)[0, :, 4, 7].cpu().numpy(), label="Input")
    for l, ls in zip(layers, line_styles):
        factor = 1
        if match_lrn_peak and "LRN" not in l.name and "LRN" in tensor_outs.keys():
            factor = tensor_outs["LRN"][:, 31] / tensor_outs[l.name][:, 31]

        lateral_pass_plot(tensor_outs[l.name] * factor, label=str(l), line_style=ls)

    plt.title(f"Effects of Different Lateral Connectivity Strategies ")
    plt.legend()
    plt.savefig(f"../documentation/figures/strategy_effects{'_match_lrn' if match_lrn_peak else ''}.pdf", format="pdf")
    plt.show()
