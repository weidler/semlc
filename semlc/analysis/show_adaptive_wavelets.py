from matplotlib import pyplot as plt

from analysis.util import get_group_model_ids, load_model_by_id
from run import generate_group_handle

network = "alexnet"
dataset = "cifar10"

group_handle = generate_group_handle(network, dataset, "adaptive-semlc")
ids = get_group_model_ids(group_handle)

wavelets = []
for id in ids[:3]:
    model = load_model_by_id(id).lateral_layer
    wavelet = model.lateral_filter.cpu().detach().numpy()
    wavelets.append(wavelet)

    plt.plot(wavelet)

# plt.plot(numpy.mean(numpy.array(wavelets), axis=0))
plt.title(group_handle)
plt.show()
