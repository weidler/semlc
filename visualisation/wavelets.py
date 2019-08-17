import matplotlib.pyplot as plt

from util.weight_initialization import mexican_hat

x = list(range(100))
for width in [3, 4, 6, 8, 10]:
    y = mexican_hat(100, width).view(-1).numpy()
    plt.plot(x, y, label=f"w={width}")

plt.legend()
plt.savefig("../documentation/figures/rickerwavelet.pdf", format="pdf", bbox_inches="tight")
plt.show()
