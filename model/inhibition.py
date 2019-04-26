import torch
from torch import nn

from util import gaussian_tensor


class Inhibition(nn.Module):

    def __init__(self, scope: int, padding="zeros"):
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
        self.convolver.weight.data = gaussian_tensor.create_mexican_hat(scope, std_1=1, std_2=2)
        self.convolver.weight.data = self.convolver.weight.data.view(1, 1, -1, 1, 1)

    def forward(self, activations: torch.Tensor):
        if self.padding_strategy == "cycle":
            activations = torch.cat((
                activations[:, :, -self.scope // 2 + 1:, :, :],
                activations,
                activations[:, :, :self.scope // 2, :, :]), dim=2)

        return self.convolver(activations)


if __name__ == "__main__":
    from pprint import pprint

    tensor_in = torch.ones([1, 1, 6, 5, 5], dtype=torch.float32)
    for i in range(6):
        tensor_in[:, :, i, :, :] *= i
    inhibitor = Inhibition(10, padding="cycle")

    tensor_out = inhibitor(tensor_in).squeeze_().squeeze_()
    pprint(tensor_out)
