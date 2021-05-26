import numpy
import torch
import torch.nn.functional as F

from analysis.util import load_model_by_id
from semlc.core.weight_initialization import generate_gabor_filter_bank

import matplotlib.pyplot as plt

from utilities.image import grid_plot, grayify_rgb_filters


def get_tuning_curve(f, s) -> numpy.ndarray:
    f = torch.unsqueeze(torch.unsqueeze(f, 0), 0)
    responses = []
    for stimulus in s:
        mean_activity = torch.mean(F.conv2d(torch.unsqueeze(torch.unsqueeze(stimulus, 0), 0), f)).numpy()
        responses.append(mean_activity)

    return numpy.array(responses)


# filters_a = generate_gabor_filter_bank((7, 7), lamb=3, n_filters=64, scale=False, part="real")
# filters_b = generate_gabor_filter_bank((7, 7), lamb=4, n_filters=64, scale=False, part="real")

filters_a = torch.tensor(grayify_rgb_filters(load_model_by_id("1621998144509098").get_conv_one().weight.cpu().detach().numpy()), dtype=torch.float32)
filters_b = torch.tensor(grayify_rgb_filters(load_model_by_id("1621998142814076").get_conv_one().weight.cpu().detach().numpy()), dtype=torch.float32)

# grid_plot(filters_b)
# plt.show()

stimuli = generate_gabor_filter_bank((7, 7), lamb=2, n_filters=len(filters_a), scale=False, part="real")

fig, axs = plt.subplots(8, 8)

i = 0
tuning_curves_base = []
tuning_curves_competitor = []
for irow in range(len(axs)):
    for icol in range(len(axs[irow])):
        tuning_curve_a = get_tuning_curve(filters_a[i], stimuli)
        tuning_curve_b = get_tuning_curve(filters_b[i], stimuli)

        tuning_curve_a -= tuning_curve_a.min()
        tuning_curve_b -= tuning_curve_b.min()
        # tuning_curve_b = tuning_curve_b + (tuning_curve_a.max() - tuning_curve_b.max())

        tuning_curves_base.append(tuning_curve_a)
        tuning_curves_competitor.append(tuning_curve_b)

        axs[irow][icol].plot(range(len(filters_a)), tuning_curve_a)
        axs[irow][icol].plot(range(len(filters_b)), tuning_curve_b)
        axs[irow][icol].set_xticks([])
        axs[irow][icol].set_yticks([])
        # plt.plot(range(len(filters_a)), tuning_curve_b)

        i += 1

plt.show()

fig, axs = plt.subplots()

axs.plot(numpy.array(tuning_curves_base).mean(axis=0))
axs.plot(numpy.array(tuning_curves_competitor).mean(axis=0))

plt.show()