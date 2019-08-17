import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

with open("../final_results/mean_acc.csv", "r") as f:
    content = f.read().split("\n")[1:]
    scopes = [[float(row.split(",")[3])] for row in content if row]
    damping = [[float(row.split(",")[5])] for row in content if row]
    width = [[float(row.split(",")[4])] for row in content if row]

scope_values = [9, 27, 45, 63]
damping_values = [.1, .12, .14, .16, .2]
width_values = [3, 4, 6, 8, 10]

fig, axs = plt.subplots(1, 3)

titles = ["Scope", "Damping", "Width"]
for idx, (ranking, values) in enumerate(zip([scopes, damping, width], [scope_values, damping_values, width_values])):
    im = axs[idx].imshow(ranking, cmap="plasma", aspect="auto")

    colors = [im.cmap(im.norm(value)) for value in values]
    patches = [mpatches.Patch(color=colors[i], label=values[i]) for i in range(len(values))]

    fig.set_size_inches(5, 6)

    axs[idx].set_title(titles[idx])
    axs[idx].axis('off')
    axs[idx].legend(handles=patches, bbox_to_anchor=(0.5, -0.42), loc="lower center", borderaxespad=.0, ncol=1)
    fig.subplots_adjust(bottom=0.3)

fig.savefig('../documentation/figures/hpoptim_other.pdf', format="pdf", bbox_inches='tight')
plt.show()
