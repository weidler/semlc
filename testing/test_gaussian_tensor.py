import matplotlib.pyplot as plt
import util.gaussian_tensor as gt


test = gt.create_mexican_hat(64, 5)
print(test)
plt.plot(test.squeeze().numpy())
plt.show()