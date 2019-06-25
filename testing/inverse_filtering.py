"""
Test equivalence of de-convolution and inversion of Toeplitz matrix.

For stability reasons it is important to keep absolute values of lateral
connection small. Below I multiply the Ricker kernel by 0.12 to achieve this.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, inv
from scipy.signal import ricker, gaussian
from scipy.fftpack import fft, ifft
# %%

n_points = 5
w_ricker = 7
w_input = 4

x = np.linspace(-10,10,101)
k = np.roll(ricker(n_points, w_ricker), int(n_points / 2)+1) * .12  # Ricker kernel
W = toeplitz(k)                               # kernel -> Toeplitz
print(W)
z = gaussian(n_points, w_input)                                     # input

"""
Scenario 1 (matrix multiplication):
da/dt = -a + z + W * a

recurrent implementation
"""

dt = 1
n_steps = 5

a_1 = np.zeros(n_points)

for i in range(n_steps):
    a_1 = dt * (z + np.dot(W, a_1))

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
delta = np.zeros(101)
delta[0] = 1
a_3 = ifft(np.divide(fft(z), fft(delta-k)))

# plotting
plt.plot(a_1, label = 'recurrent')
plt.plot(a_2, '-.', label = 'toeplitz')
plt.plot(a_3, '.', label = 'deconvolution')
plt.legend()
plt.show()