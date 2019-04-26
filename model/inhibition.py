import torch

from torch import nn


class Inhibition(nn.Module):

    def __init__(self, scope: int):
        super().__init__()

        self.convolver: nn.Conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=(scope, 1, 1),
            stride=(1, 1, 1),
            padding=(scope // 2, 0, 0),
            dilation=(1, 1, 1),
            bias=0
        )

        self.convolver.weight.data = torch.ones([1, 1, scope, 1, 1])/scope

    def forward(self, activations: torch.Tensor):
        return self.convolver(activations)


if __name__ == "__main__":
    from pprint import pprint

    tensor_in = torch.ones([1, 1, 6, 5, 5], dtype=torch.float32)
    inhibitor = Inhibition(5)

    tensor_out = inhibitor(tensor_in).squeeze_().squeeze_()
    pprint(tensor_out)