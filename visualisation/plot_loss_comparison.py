import statistics

import matplotlib.pyplot as plt

with open("../results/BaseClassificationCNN.loss", "r") as f:
    base_loss = [float(t.split("\t")[1]) for t in f.readlines()]

with open("../results/InhibitionClassificationCNN.loss", "r") as f:
    single_loss = [float(t.split("    ")[1]) for t in f.readlines()]

with open("../results/RecInhibitionClassificationCNNmy um.loss", "r") as f:
    rec_loss = [float(t.split("\t")[1]) for t in f.readlines()]

base_loss = [statistics.mean(base_loss[i-2:i+1]) for i in range(2, len(base_loss), 3)]
single_loss = [statistics.mean(single_loss[i-2:i+1]) for i in range(2, len(single_loss), 3)]
rec_loss = [statistics.mean(rec_loss[i-2:i+1]) for i in range(2, len(rec_loss), 3)]
x = list(range(1, len(base_loss) + 1))
print(x)
plt.xlim(1, len(base_loss))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(x, base_loss, label="Baseline")
plt.plot(x, single_loss, label="Single-Shot")
plt.plot(x, rec_loss, label="Recurrent")
plt.legend()

plt.show()