import random

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from torch import Tensor, nn


class ActivationVisualization:

    fig: Figure

    def __init__(self):
        self.fig, self.axs = plt.subplots()
        self.fig.suptitle("Original Activations and their Inhibitions")

    def visualize(self, activation: Tensor, inhibition: Tensor):
        activation = activation[0, 0, :, 0, 0].tolist()
        inhibition = inhibition[0, 0, :, 0, 0].tolist()

        width = 0.8
        margin = 0.3

        self.axs.bar([i * width * 2 - width + i * margin for i in range(len(activation))], activation, width, label="Input")
        self.axs.bar([i * width * 2 + i * margin for i in range(len(activation))], inhibition, width, label="Inhibition")

        self.axs.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.axs.legend()

        plt.show()


if __name__ == "__main__":
    a = Tensor([random.randint(3, 10) for i in range(10)])
    b = Tensor([random.randint(5, 12) for i in range(10)])

    o = ActivationVisualization()
    o.visualize(a, b)
