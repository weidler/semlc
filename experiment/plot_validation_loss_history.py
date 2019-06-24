import os
import re

import numpy
import matplotlib.pyplot as plt

compared_models = ["baseline", "cmapbaseline", "ss_freeze"]

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
    plt.plot(mean_histories[model_name], label=model_name)

plt.legend()
plt.show()