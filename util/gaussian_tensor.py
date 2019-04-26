import torch

def create_distributed_tensor(num_filters, num_samples=1000000, mean=1.0, std=0.0):
    normal = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([std]))
    samples = normal.sample((num_samples,))
    hist = torch.histc(samples, bins=num_filters)/(num_samples/float(num_filters))
    distr_tensor = hist.unsqueeze(0).unsqueeze(0)
    return distr_tensor
