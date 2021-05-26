import numpy
import torch
import torch.nn.functional as F

from analysis.util import load_model_by_id
from networks import BaseNetwork
from semlc.core.weight_initialization import generate_gabor_filter_bank

import matplotlib.pyplot as plt

from utilities.image import grid_plot, grayify_rgb_filters


def get_tuning_curves(model: BaseNetwork, s) -> numpy.ndarray:
    gray_filters = torch.unsqueeze(torch.tensor(grayify_rgb_filters(model.get_conv_one().weight.cpu().detach().numpy()),
                                                dtype=torch.float32), 1)
    stimuli = torch.unsqueeze(torch.stack(s, 0), 1)

    out = F.conv2d(stimuli, gray_filters, padding=model.get_conv_one().padding)
    if model.is_lateral:
        out = model.lateral_layer(out)

    return F.relu(out).numpy()


model_base = load_model_by_id("1622055130772184")
model_competitor = load_model_by_id("1622058944269511")
n_filters = model_base.get_conv_one().out_channels

stimuli = generate_gabor_filter_bank((7, 7), lamb=2, n_filters=n_filters, scale=False, part="real")
tuning_curves_base = get_tuning_curves(model_base, stimuli).mean(-1).mean(-1)
tuning_curves_competitor = get_tuning_curves(model_competitor, stimuli).mean(-1).mean(-1)

rolled_tuning_curves_competitor = []
rolled_tuning_curves_base = []

fig, axs = plt.subplots(8, 8)

i = 0
for irow in range(len(axs)):
    for icol in range(len(axs[irow])):
        tuning_curve_base = tuning_curves_base[:, i, ...]
        tuning_curve_competitor = tuning_curves_competitor[:, i, ...]

        tuning_curve_base -= tuning_curve_base.min()
        tuning_curve_competitor -= tuning_curve_competitor.min()

        base_peak_index = numpy.argmax(tuning_curve_base)
        competitor_peak_index = numpy.argmax(tuning_curve_competitor)

        rolled_tuning_curves_base.append(numpy.roll(tuning_curve_base, len(tuning_curve_base) - base_peak_index + len(tuning_curve_base) // 2))
        rolled_tuning_curves_competitor.append(numpy.roll(tuning_curve_competitor, len(tuning_curve_competitor) - competitor_peak_index + len(tuning_curve_competitor) // 2))

        axs[irow][icol].plot(range(n_filters), tuning_curve_base)
        axs[irow][icol].plot(range(n_filters), tuning_curve_competitor)
        axs[irow][icol].set_xticks([])
        axs[irow][icol].set_yticks([])
        # plt.plot(range(n_filters), tuning_curve_b)

        i += 1

fig.show()

fig, axs = plt.subplots()

rolled_tuning_curves_base = numpy.array(rolled_tuning_curves_base)
rolled_tuning_curves_competitor = numpy.array(rolled_tuning_curves_competitor)
mean_tc_base = rolled_tuning_curves_base.mean(axis=0)
mean_tc_competitor = rolled_tuning_curves_competitor.mean(axis=0)

# mean_tc_base = mean_tc_base * (mean_tc_competitor.max() / mean_tc_base.max())

axs.plot(mean_tc_base - mean_tc_base.min(), label="No SemLC")
axs.plot(mean_tc_competitor - mean_tc_competitor.min(), label="SemLC")

axs.legend()
plt.show()
