import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with open("../results/hpopt.csv", "r") as f:
    ranking = [row.split(",")[2] for row in f.read().split("\n")[1:] if row]

KEY_MAP = {
    "converged": 2,
    "toeplitz": 1,
    "once": 3,
    "once_learned": 4
}

VALUE_MAP = {
    1: "Converged Frozen",
    2: "Converged Adaptive",
    3: "Single Shot Frozen",
    4: "Single Shot Adaptive"
}

mapped_ranking = [[KEY_MAP[e]] for e in ranking]

colors = ["#c4391d", "#e08600", "#0c72cc", "#09dae5"]
cmp = ListedColormap(colors)

im = plt.imshow(mapped_ranking, cmap=cmp, aspect="auto")
values = [1, 2, 3, 4]

# colormap used by imshow
colors = [im.cmap(im.norm(value)) for value in values]
# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=colors[i], label=VALUE_MAP[i + 1]) for i in range(len(values))]

plt.gcf().set_size_inches(5, 6)

plt.axis('off')
plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.22), loc="lower center", borderaxespad=.0, ncol=2)
plt.gcf().subplots_adjust(bottom=0.2, left=0, right=0.68)
plt.savefig('../documentation/figures/hpoptim.pdf', format="pdf", bbox_inches='tight')
