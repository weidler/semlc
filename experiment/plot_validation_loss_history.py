import os
import re
from typing import List

import numpy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

compared_models = ["baseline", "cmap", "ss_freeze", "ss", "converged_freeze", "converged"]
model_names = ["Baseline", "Baseline + LRN", "Single Shot Frozen",
               "Single Shot Adaptive", "Converged Frozen", "Converged Adaptive"]
# compared_models = ["baseline", "converged_freeze"]

axs: List[Axes]
fig: Figure
fig, axs = plt.subplots(2, 1)

histories = {}
mean_histories = {}
for model_name in compared_models:
    histories.update({model_name: []})
    for i in range(1, 11):
        path = f"../final_results/{model_name}/{model_name}_{i}/"
        file_names = os.listdir(path)
        accuracy_file_name = [n for n in file_names if re.match(".*\.acc", n)][0]

        with open(path + accuracy_file_name, "r") as f:
            accuracies = [float(row.split("\t")[1]) for row in f.read().split("\n") if row != ""]
            if len(accuracies) == 160:
                histories[model_name].append(accuracies)
            else:
                print("[WARNING]: Unexpected number of epochs.")
    print(len(histories[model_name]))
    mean_histories.update({model_name: numpy.mean(numpy.array(histories[model_name]), axis=0)})
    axs[0].plot(mean_histories[model_name], label=model_names[compared_models.index(model_name)])
    axs[1].plot(mean_histories[model_name], label=model_names[compared_models.index(model_name)])

axs[1].set_ylim(78, 84)
axs[1].set_xlim(80, 160)

axs[0].set_xlabel("Epoch")
axs[1].set_xlabel("Epoch")
axs[0].set_ylabel("Validation Accuracy")
axs[1].set_ylabel("Validation Accuracy")

fig.set_size_inches(4.5, 9)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2)
fig.subplots_adjust(hspace=0.35, bottom=0.17)

fig.savefig("../documentation/figures/acc_history.pdf", format="pdf", bbox_inches='tight')

plt.show()