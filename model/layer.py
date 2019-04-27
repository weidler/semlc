import torch
from torch import nn

from util import weight_initialization


class Inhibition(nn.Module):
    """Nice Inhibition Layer. """

    def __init__(self, scope: int, padding: str="zeros", learn_weights=False):
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
        if activations.shape[2] < self.scope:
            raise RuntimeError("Inhibition not possible. "
                               "Given activation has less filters than the Inhibitor's scope.")

        # augment channel dimension
        activations.unsqueeze_(dim=1)

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



if __name__ == "__main__":
    from pprint import pprint

    scope = 15
    tensor_in = torch.ones([1, 1, scope, 5, 5], dtype=torch.float32)
    for i in range(tensor_in.shape[2]):
        tensor_in[:, :, i, :, :] *= i
    inhibitor = Inhibition(scope, padding="zeros")

    for i in range(9898):
        tensor_out = inhibitor(tensor_in).squeeze_().squeeze_()
    pprint(tensor_out)
