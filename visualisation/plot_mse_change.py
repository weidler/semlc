import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text


def auto_label(rects, labels, axis: Axes):
    """Attach a text label above each bar displaying its height. Adapted from the matplotlib docs."""
    for label, rect in zip(labels, rects):
        height = rect.get_height()

        y_pos = 1.05 * height
        t: Text = axis.text(rect.get_x() + rect.get_width() / 2., y_pos,
                  str(round(label, 3)),
                  ha='center', va='bottom')
        if height < 0:
            t.set_y(t.get_position()[1] - 0.035)


changes = (100 - np.array([100.36, 99.59, 74.49, 79.90, 61.28, 77.83, 60.00])) / 100
strategies = ["Base. Alexnet",
              "Base. AlexCMap",
              "SSI Frozen",
              "SSI Adaptive",
              "Conv. Frozen",
              "Conv. Adaptive",
              "Conv. Parametric"]

bc, ssi_c, conv_c = plt.get_cmap("Set1").colors[:3]
colors = [bc, bc, ssi_c, ssi_c, conv_c, conv_c, conv_c]

ax: Axes
fig: Figure
fig, ax = plt.subplots()

bars = ax.bar(strategies, changes, color=colors)
auto_label(bars, labels=changes, axis=ax)
ax.set_ylabel("MSE Ratio")

ax.axhline(y=0, color='k')
ax.set_xticklabels(strategies, rotation="vertical")
ax.margins(0.05, 0.2)
fig.subplots_adjust(bottom=0.3)
# fig.suptitle("MSE Ratio of All Filter Pairs Over Adjacent Filter Pairs")

plt.savefig("../documentation/figures/mse-ratio-change.pdf", format="pdf")
plt.show()
