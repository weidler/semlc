import random

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal.windows import gaussian
from torch import nn

from layers import SemLC, GaussianSemLC, LRN

lw = 1
ms = 3


def lateral_pass_plot(tensor_out, label, line_style="."):
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), line_style, label=label, lw=lw, ms=ms)


def gaussian_input_signal():
    tensor_in = torch.zeros((batches, depth, width, height))
    for b in range(batches):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(
                    gaussian(depth, wavelet_width)
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), -(scope // 4)) * 0.2
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), (scope // 4)) * 0.2
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), -(scope // 2)) * 0.1
                    + np.roll(gaussian(depth, wavelet_width * random.uniform(0, 3)), (scope // 2)) * 0.1)

    return tensor_in * 100


def random_input_signal():
    tensor_in = torch.rand((batches, depth, width, height))
    return tensor_in * 100


def two_proximal_hypotheses_signal(distance):
    tensor_in = torch.zeros((batches, depth, width, height))
    for b in range(batches):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(
                    + np.roll(gaussian(depth, wavelet_width), -(scope // distance))
                    + np.roll(gaussian(depth, wavelet_width), (scope // distance)))

    return tensor_in * 100


if __name__ == "__main__":
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["font.weight"] = 'normal'

    matplotlib.rcParams["font.size"] = 9
    matplotlib.rcParams["legend.fontsize"] = 9
    matplotlib.rcParams["axes.labelsize"] = 9
    matplotlib.rcParams["axes.titlesize"] = 9
    matplotlib.rcParams["xtick.labelsize"] = 9
    matplotlib.rcParams["ytick.labelsize"] = 9

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
    wavelet_width = 2
    damping = 0.2
    self_connect = False
    match_lrn_peak = False

    input_tensors = [
        ("two_peak_16", two_proximal_hypotheses_signal(16)),
        ("two_peak_14", two_proximal_hypotheses_signal(14)),
        ("two_peak_12", two_proximal_hypotheses_signal(12)),
        ("two_peak_10", two_proximal_hypotheses_signal(10)),
        ("two_peak_8", two_proximal_hypotheses_signal(8)),
        ("two_peak_6", two_proximal_hypotheses_signal(6)),
        ("two_peak_4", two_proximal_hypotheses_signal(4)),
        ("two_peak_2", two_proximal_hypotheses_signal(2)),
        ("random", random_input_signal()),

    ]

    for input_name, tensor_in in input_tensors:

        simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)

        layers = [
            SemLC(simple_conv, widths=wavelet_width, damping=damping, self_connection=self_connect),
            GaussianSemLC(simple_conv, widths=wavelet_width, damping=damping, self_connection=self_connect),
            LRN(simple_conv, widths=wavelet_width, damping=damping)]

        line_styles = ["-", "--", ".-", "."]
        line_styles += ["-" for i in range(len(layers) - len(line_styles))]

        tensor_outs = {l.name: l(tensor_in) for l in layers}

        fig: Figure
        axs: Axes
        fig, axs = plt.subplots()

        input_factor_lrn = (tensor_outs["LRN"][:, 31] / tensor_in[:, 31]) if match_lrn_peak else 1
        axs.plot((tensor_in)[0, :, 4, 7].cpu().numpy(), ":", label="Input", lw=lw, ms=ms)
        for l, ls in zip(layers, line_styles):
            factor = 1
            if match_lrn_peak and "LRN" in tensor_outs.keys() and "LRN" in l.name:
                factor = tensor_in[:, 31] / tensor_outs[l.name][:, 31]

            lateral_pass_plot(tensor_outs[l.name] * factor, label=l.name, line_style=ls)

        axs.legend()

        axs.set_xlim(16, depth - 16)

        axs.set_xlabel("Filter")
        axs.set_ylabel("Activation")

        fig.set_size_inches(6.75, 2.5)
        fig.tight_layout()

        plt.savefig(f"./strategy_effects{'_match_lrn' if match_lrn_peak else ''}_{input_name}.pdf",
                    format="pdf", dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
