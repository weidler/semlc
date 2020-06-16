import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from model.fft_inhibition_layer import FFTConvergedFrozenInhibition, FFTConvergedInhibition
from model.inhibition_layer import SingleShotInhibition, ConvergedInhibition, \
    ConvergedFrozenInhibition, ParametricInhibition, SingleShotGaussian, ConvergedGaussian


def lateral_pass_plot(layer, signal, line_style="."):
    tensor_out = layer(signal)
    plt.plot(tensor_out[0, :, 4, 7].detach().cpu().numpy(), line_style, label=layer.name)


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
    scope = 99
    depth = 100
    width = 14
    height = 14
    wavelet_width = 6
    damping = 0.12
    self_connect = False

    tensor_in = torch.zeros((batches, depth, width, height))
    for b in range(batches):
        for i in range(tensor_in.shape[-1]):
            for j in range(tensor_in.shape[-2]):
                tensor_in[b, :, i, j] = torch.from_numpy(
                    gaussian(depth, 6)
                    + np.roll(gaussian(depth, 6), -(scope // 4)) * 0.5
                    + np.roll(gaussian(depth, 6), (scope // 4)) * 0.5
                    + np.roll(gaussian(depth, 6), -(scope // 2)) * 0.2
                    + np.roll(gaussian(depth, 6), (scope // 2)) * 0.2
                )

    tensor_in *= 100

    simple_conv = nn.Conv2d(depth, depth, 3, 1, padding=1)

    layers = [
        SingleShotInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, learn_weights=True, pad="zeros",
                             self_connection=self_connect),
        SingleShotInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, learn_weights=True, pad="circular",
                             self_connection=self_connect),

        # circular padding
        ConvergedInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, self_connection=self_connect),
        ConvergedFrozenInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping,
                                  self_connection=self_connect),
        ParametricInhibition(in_channels=depth, ricker_width=wavelet_width, initial_damp=damping,
                             self_connection=self_connect),

        FFTConvergedInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping),
        FFTConvergedFrozenInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping),

        # zero padding
        # ConvergedInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, pad="zeros", self_connection=self_connect),
        # ConvergedFrozenInhibition(in_channels=depth, ricker_width=wavelet_width, in_channels=depth, damp=damping, pad="zeros", self_connection=self_connect),

        ConvergedGaussian(in_channels=depth, ricker_width=wavelet_width, damp=damping, self_connection=self_connect),
        SingleShotGaussian(in_channels=depth, ricker_width=wavelet_width, damp=damping, pad="circular", self_connection=self_connect),
        # RecurrentInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, self_connection=self_connect),
        # RecurrentInhibition(in_channels=depth, ricker_width=wavelet_width, damp=damping, self_connection=self_connect),
    ]

    line_styles = [".", "-", "-", "--", ".-", "."]
    line_styles += ["." for i in range(len(layers) - len(line_styles))]

    plt.clf()
    plt.plot(tensor_in[0, :, 4, 7].cpu().numpy(), label="Input")
    [lateral_pass_plot(l, tensor_in, line_style=ls) for l, ls in zip(layers, line_styles)]

    plt.title(f"Effects of Different Inhibition Strategies ")
    plt.legend()
    plt.savefig(f"../documentation/figures/strategy_effects.pdf", format="pdf")
    plt.show()
