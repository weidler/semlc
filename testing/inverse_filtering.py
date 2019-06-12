"""
Test equivalence of de-convolution and inversion of Toeplitz matrix.

For stability reasons it is important to keep absolute values of lateral
connection small. Below I multiply the Ricker kernel by 0.12 to achieve this.
"""
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, inv
from scipy.signal import ricker, gaussian
from scipy.fftpack import fft, ifft
# %%
from util.complex import div_complex

n_points = 101
w_ricker = 7
w_input = 4

x = np.linspace(-10,10,101)
k = np.roll(ricker(n_points, w_ricker), int(n_points / 2)+1) * .12  # Ricker kernel
W = toeplitz(k)                                                     # kernel -> Toeplitz
z = gaussian(n_points, w_input)                                     # input

"""
Scenario 1 (matrix multiplication):
da/dt = -a + z + W * a

recurrent implementation
"""

dt = 1
n_steps = 500

a_1 = np.zeros(n_points)

for i in range(n_steps):
    a_1 += dt * (-a_1 + z + np.dot(W, a_1))

"""
Scenario 2 (matrix multiplication):
da/dt = -a + z + W * a

implement by setting da/dt to 0 and solving for a:
a = (I - W)^-1 z 

"""
a_2 = np.dot(inv(np.eye(n_points) - W), z)

"""
Scenario 3 (convolution):
da/dt = -a + z + conv(a,k) 

implement by setting da/dt to 0 and solving for a:
a = deconv((delta - k),z) 

"""


delta = torch.zeros(101).type(dtype=torch.double)
z = torch.from_numpy(z).type(dtype=torch.double)
k = torch.from_numpy(k).type(dtype=torch.double)
delta[0] = 1
fourier_z = torch.rfft(z, 1, onesided=False)
fourier_convoluted = div_complex(fourier_z, torch.rfft(delta-k, 1, onesided=False))
# fourier_convoluted = fourier_convoluted.where(~torch.isnan(fourier_convoluted), fourier_z)
a_3 = torch.irfft(fourier_convoluted, 1, onesided=False)
print(a_3)
a_3 = a_3.numpy()

# plotting
plt.plot(a_1, label = 'recurrent')
plt.plot(a_2, '-.', label = 'toeplitz')
plt.plot(a_3, '.', label = 'deconvolution')
plt.legend()
plt.show()