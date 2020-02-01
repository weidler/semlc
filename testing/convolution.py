import numpy
import torch

from model.alternative_inhibition_layers import pad_shift_zero
from util.convolution import toeplitz1d_circular, toeplitz1d_zero, pad_roll

signal = torch.tensor([1, 2, 3, 4, 5, 4, 3, 2, 1])
cfilter = torch.tensor([2, 2, 2])

tpl_circular = toeplitz1d_circular(cfilter, 9)
tpl_zeros = toeplitz1d_zero(cfilter.squeeze(), 9)

print(tpl_zeros.transpose(1, 0))
print(f"Target: \t{[6, 12, 18, 24, 26, 24, 18, 12, 6]}")
print(f"Circular: \t{signal.matmul(tpl_circular).tolist()}")
print(f"ZeroPad: \t{signal.matmul(tpl_zeros).tolist()}")

# print(tpl_zeros)
