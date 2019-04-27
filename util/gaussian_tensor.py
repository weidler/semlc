import torch
from scipy import signal


def create_distributed_tensor_old(scope: int, num_samples: int = 1000000, mean: float = 1.0,
                                  std: float = 0.5) -> torch.Tensor:
    normal = torch.distributions.Normal(torch.tensor([mean], dtype=torch.float32),
                                        torch.tensor([std], dtype=torch.float32))
    samples = normal.sample((num_samples,))
    hist = torch.histc(samples, bins=scope) / (num_samples / float(scope))
    distr_tensor = hist.unsqueeze(0).unsqueeze(0)
    return distr_tensor


def create_mexican_hat(scope: int, std: float) -> torch.Tensor:
    window = signal.ricker(scope, std)
    distr_tensor = torch.Tensor([[window]])

    return distr_tensor
