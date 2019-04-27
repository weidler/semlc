import copy

import torch
from torch import nn

from util import weight_initialization


class Inhibition(nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, padding: str = "zeros", learn_weights=False):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.padding_strategy = padding
        self.scope = scope

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
        activations = self.convolver(activations)

        # return inhibited activations without augmented channel dimension
        return activations.squeeze_(dim=1)


class RecurrentInhibition(nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, padding: str = "zeros", learn_weights: bool = False, decay: float = 0.05,
                 max_steps: int = 20):
        super().__init__()

        assert padding in ["zeros", "cycle"]
        self.max_steps = max_steps
        self.decay = decay
        self.padding_strategy = padding
        self.scope = scope

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
        inhib_in: torch.Tensor = self.W_in(activations)

        steps = 0
        converged_inhibition: torch.Tensor = inhib_in.clone()
        while steps < self.max_steps:
            inhib_rec = self.W_rec(converged_inhibition)
            phi = (inhib_in + inhib_rec) / 2

            converged_inhibition = (1 - self.decay) * converged_inhibition + self.decay * phi

            steps += 1

            # x = list(range(self.scope))
            #
            # plt.cla()
            # plt.plot(x, [f.item() for f in (activations / activations.sum())[0, 0, :, 0, 0]], label="Original")
            # plt.plot(x, [f.item() for f in (converged_inhibition / converged_inhibition.sum())[0, 0, :, 0, 0]], label="Recurrent")
            # plt.legend()
            # plt.pause(0.001)

        # return inhibited activations without augmented channel dimension
        return converged_inhibition.squeeze_(dim=1)


if __name__ == "__main__":

    scope = 15
    tensor_in = torch.ones([1, scope, 5, 5], dtype=torch.float32)
    for i in range(tensor_in.shape[1]):
        tensor_in[:, i, :, :] *= i
    inhibitor = Inhibition(scope, padding="zeros")
    inhibitor_rec = RecurrentInhibition(scope, padding="zeros", max_steps=1000000)
    print(tensor_in.shape)

    for i in range(1):
        tensor_out = inhibitor(tensor_in)
        tensor_out_rec = inhibitor_rec(tensor_in)
