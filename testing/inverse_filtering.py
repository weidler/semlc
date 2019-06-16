"""
Test equivalence of de-convolution and inversion of Toeplitz matrix.

For stability reasons it is important to keep absolute values of lateral
connection small. Below I multiply the Ricker kernel by 0.12 to achieve this.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import toeplitz, inv
from scipy.signal import ricker, gaussian

from util import weight_initialization
from util.complex import div_complex

n_points = 101
w_ricker = 7
w_input = 4

k = weight_initialization.mexican_hat(n_points, std=w_ricker).numpy() * .12
k = np.roll(ricker(n_points, w_ricker), int(n_points / 2) + 1) * .12  # Ricker kernel
W = toeplitz(k)  # kernel -> Toeplitz
z = gaussian(n_points, w_input)  # input

"""
Scenario 1 (matrix multiplication):
da/dt = -a + z + W * a

recurrent implementation
"""

dt = 1
n_steps = 5

a_1 = np.zeros(n_points)

for i in range(n_steps):
    a_1 += dt * (-a_1 + z + np.dot(W, a_1))

"""
Scenario 2 (matrix multiplication):
da/dt = -a + z + W * a

implement by setting da/dt to 0 and solving for a:
a = (I - W)^-1 z 

"""
a_2 = np.matmul(z, inv(np.eye(n_points) - W))

"""
Scenario 3 (convolution):
da/dt = -a + z + conv(a,k) 

implement by setting da/dt to 0 and solving for a:
a = deconv((delta - k),z) 

"""

#
# delta = torch.zeros(101).type(dtype=torch.double)
# z = torch.from_numpy(z).type(dtype=torch.double)
# k = torch.from_numpy(k).type(dtype=torch.double)
# delta[0] = 1
# fourier_z = torch.rfft(z, 1, onesided=False)
# fourier_convoluted = div_complex(fourier_z, torch.rfft(delta - k, 1, onesided=False))
#
# a_3 = torch.irfft(fourier_convoluted, 1, onesided=False)
# a_3 = a_3.numpy()


# TENSOR deconvolution
# INPUT FORMAT IS N x C x H x W

n = 1
channels = 101
height = 14
width = 14

z_tensor = torch.zeros((n, channels, height, width)).type(dtype=torch.double)
for i in range(width):
    for j in range(height):
        z_tensor[0, :, i, j] = torch.from_numpy(gaussian(101, w_input))
# MUST use permute not view as it preserves data organization better
z_tensor = z_tensor.permute((0, 2, 3, 1))

delta = torch.zeros(101).type(dtype=torch.double)
delta[0] = 1
delta = delta.view((1, 1, 1, -1))
k = torch.from_numpy(k).type(dtype=torch.double)
k = k.view((1, 1, 1, -1))

fourier_z = torch.rfft(z_tensor, 1, onesided=False)
fourier_filter = torch.rfft(delta - k, 1, onesided=False)
fourier_convoluted = div_complex(fourier_z, fourier_filter)

inhibited_tensor = torch.irfft(fourier_convoluted, 1, onesided=False)
inhibited_tensor = inhibited_tensor.permute((0, 3, 1, 2))
a_4 = inhibited_tensor[0, :, random.randint(0, 13), random.randint(0, 13)].numpy()


# plotting
plt.plot(a_1, label='recurrent')
plt.plot(a_2, '-.', label='toeplitz')
# plt.plot(a_3, '.', label='deconvolution')
plt.plot(a_4, '--', label='deconvolution tensor')
plt.legend()
plt.show()
