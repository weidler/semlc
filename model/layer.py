import copy
import math
import random
from typing import List

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch import nn

from util import weight_initialization
from visualisation.analysis import ActivationVisualization


class Inhibition(nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, padding: str = "zeros", learn_weights=False, analyzer=None):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.padding_strategy = padding
        self.scope = scope
        self.analyzer = analyzer

        self.convolver: nn.Conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(scope, 1, 1),
            stride=(1, 1, 1),
            padding=(scope // 2, 0, 0) if padding == "zeros" else (0, 0, 0),
            dilation=(1, 1, 1),
            bias=0
        )

        # apply gaussian
        self.convolver.weight.data = weight_initialization.mexican_hat(scope, std=2)
        self.convolver.weight.data = self.convolver.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.convolver.parameters():
                print(param.requires_grad)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        # catch weird scope-input-combinations; TODO do we really want this?
        if activations.shape[1] < self.scope:
            raise RuntimeError("Inhibition not possible. "
                               "Given activation has less filters than the Inhibitor's scope.")

        # augment channel dimension
        activations = activations.unsqueeze(dim=1)

        # apply cycle padding strategy if necessary
        if self.padding_strategy == "cycle":
            activations = torch.cat((
                activations[:, :, -self.scope // 2 + 1:, :, :],
                activations,
                activations[:, :, :self.scope // 2, :, :]), dim=2)

        # inhibit
        inhibitions = self.convolver(activations)

        # analyse
        if self.analyzer is not None:
            self.analyzer.visualize(activations, inhibitions)

        # return inhibited activations without augmented channel dimension
        return inhibitions.squeeze_(dim=1)


class RecurrentInhibition(nn.Module):
    """Nice Inhibition Layer. """
    axs_convergence: List[Axes]
    fig_convergence: plt.Figure

    def __init__(self, scope: int, padding: str = "zeros", learn_weights: bool = False, decay: float = 0.05,
                 max_steps: int = 20, convergence_threshold: float = 0.001):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.max_steps = max_steps
        self.decay = decay
        self.padding_strategy = padding
        self.scope = scope
        self.convergence_threshold = convergence_threshold

        self.W_in: nn.Conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(scope, 1, 1),
            stride=(1, 1, 1),
            padding=(scope // 2, 0, 0) if padding == "zeros" else (0, 0, 0),
            dilation=(1, 1, 1),
            bias=0
        )

        # apply gaussian
        self.W_in.weight.data = weight_initialization.mexican_hat(scope, std=2)
        self.W_in.weight.data = self.W_in.weight.data.view(1, 1, -1, 1, 1)

        # freeze weights if desired to retain initialized structure
        if not learn_weights:
            for param in self.W_in.parameters():
                print(param.requires_grad)

        # recurrent convolution as a copy of input convolution
        self.W_rec: nn.Conv3d = copy.deepcopy(self.W_in)

        # figures for visualization
        self.fig_convergence, self.axs_convergence = plt.subplots(1, 2)
        self.fig_convergence.set_size_inches(12, 5)
        self.fig_convergence.suptitle("Convergence of Single Recursive Inhibition Process", )

    def forward(self, activations: torch.Tensor, plot_convergence=False) -> torch.Tensor:
        # catch weird scope-input-combinations; TODO do we really want this?
        if activations.shape[1] < self.scope:
            raise RuntimeError("Inhibition not possible. "
                               "Given activation has less filters than the Inhibitor's scope.")

        # augment channel dimension
        activations = activations.unsqueeze(dim=1)

        # apply cycle padding strategy if necessary
        if self.padding_strategy == "cycle":
            activations = torch.cat((
                activations[:, :, -self.scope // 2 + 1:, :, :],
                activations,
                activations[:, :, :self.scope // 2, :, :]), dim=2)

        # inhibit
        inhib_in: torch.Tensor = self.W_in(activations)

        steps = 0
        step_difference = math.inf
        step_differences = []
        converged_inhibition: torch.Tensor = inhib_in.clone()

        while steps < self.max_steps and step_difference > self.convergence_threshold:
            inhib_rec = self.W_rec(converged_inhibition)
            phi = (inhib_in + inhib_rec) / 2

            previous_converged_inhibition = converged_inhibition
            converged_inhibition = (1 - self.decay) * converged_inhibition + self.decay * phi

            steps += 1
            step_difference = nn.functional.mse_loss(previous_converged_inhibition, converged_inhibition).item()
            step_differences.append(step_difference)

            if plot_convergence:
                self._plot_inhibition_convergence(activations, inhib_in, converged_inhibition, step_differences)

        # return inhibited activations without augmented channel dimension
        return converged_inhibition.squeeze_(dim=1)

    def _plot_inhibition_convergence(self, signal_in: torch.Tensor, signal_inhib: torch.Tensor,
                                     signal_rec: torch.Tensor, step_differences: List[float]) -> None:

        self.axs_convergence[0].cla(), self.axs_convergence[1].cla()
        self.axs_convergence[1].set_xlim(0, self.max_steps), self.axs_convergence[1].set_ylim(0, 0.03)

        x = list(range(self.scope))
        self.axs_convergence[0].plot(x, [f.item() for f in signal_in[0, 0, :, 0, 0]], label="Original")
        self.axs_convergence[0].plot(x, [f.item() for f in signal_inhib[0, 0, :, 0, 0]], label="Single Step")
        self.axs_convergence[0].plot(x, [f.item() for f in signal_rec[0, 0, :, 0, 0]], label="Recurrent")

        self.axs_convergence[1].plot(step_differences, label="Difference to last step")
        self.axs_convergence[1].plot([self.convergence_threshold for _ in list(range(self.max_steps))],
                                     label="Convergence Threshold", color="red")

        self.axs_convergence[0].legend(), self.axs_convergence[1].legend()

        plt.pause(0.001)


if __name__ == "__main__":

    scope = 15
    tensor_in = torch.ones([1, scope, 5, 5], dtype=torch.float32)
    for i in range(tensor_in.shape[1]):
        tensor_in[:, i, :, :] *= random.randint(1, tensor_in.shape[1])
    inhibitor = Inhibition(scope, padding="zeros", analyzer=ActivationVisualization())
    inhibitor_rec = RecurrentInhibition(scope, padding="zeros", max_steps=1000)

    for i in range(100):
        tensor_out = inhibitor(tensor_in)
        tensor_out_rec = inhibitor_rec(tensor_in, plot_convergence=False)
