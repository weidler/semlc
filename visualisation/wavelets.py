import matplotlib.pyplot as plt
import numpy

from util.weight_initialization import mexican_hat

x = numpy.array(list(range(99))) - 49
for width in [2, 5, 10, 14]:
    y = mexican_hat(99, width).view(-1).numpy()
    plt.plot(x, y, label=f"\u03C3={width}")

plt.xlabel("Relative Filter Position")
plt.ylabel("Weight")

plt.legend()
plt.savefig("../documentation/figures/wavelet.pdf", format="pdf", bbox_inches="tight")
plt.show()
