import matplotlib.pyplot as plt
import gaussian_tensor as gt


test = gt.create_distributed_tensor(32, mean=4.0, std=0.5)
print(test)
plt.plot(test.squeeze().numpy())
plt.show()