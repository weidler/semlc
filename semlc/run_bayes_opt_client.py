import argparse
import json
import time
from typing import Any, Dict

import numpy
import pandas as pd
from ax.plot.render import plot_config_to_html
from ax.service.ax_client import AxClient
from ax.utils.report.render import render_report_elements
from mpi4py import MPI
from scipy import stats

import run
from networks.util import AVAILABLE_NETWORKS
from utilities.data import AVAILABLE_DATASETS
from utilities.util import HiddenPrints

comm = MPI.COMM_WORLD
rank = comm.rank
is_root = rank == 0

hpoptim_id = int(time.time())


def train_evaluate(p: Dict[str, Any], other_args):
    other_args["widths"] = (p["w1"], p["w2"])
    other_args["damps"] = p["d"]
    other_args["ratio"] = p["r"]

    if not is_root:
        with HiddenPrints():
            evaluation_results = run.run(other_args, verbose=False)
    else:
        evaluation_results = run.run(other_args, verbose=False)

    return evaluation_results["default"]["total"], evaluation_results["percent_less_chaos"]


def evaluate(p: Dict[str, Any], other_args) -> Dict:
    process_acc, order = train_evaluate(p, other_args)
    gathered_means, gathered_order_means = comm.allgather(process_acc), comm.allgather(order)

    if len(gathered_means) > 1:
        mean, sem = numpy.mean(gathered_means), stats.sem(gathered_means)
        order_mean, order_sem = numpy.mean(gathered_order_means), stats.sem(gathered_order_means)
    else:
        mean, sem = gathered_means[0], 0.0
        order_mean, order_sem = gathered_order_means[0], 0.0

    return {"accuracy": (mean, sem), "order": (order_mean, order_sem)}


# PARSE CLIENT PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument("network", type=str, choices=AVAILABLE_NETWORKS, help="The network for which to hyperoptimize")
parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations the BO evaluates.")
parser.add_argument("--initial", type=int, default=5, help="Number of initial configs produced to setup BO.")
parser.add_argument("-e", "--epochs", type=int, default=40, help="Number of epochs per model.")
parser.add_argument("--metric", type=str, choices=["accuracy", "order"], default="accuracy", help="Used metric.")

# peripheral, optimization related arguments
parser.add_argument("--data", type=str, default="cifar10", choices=AVAILABLE_DATASETS, help="dataset to use")
parser.add_argument("--rings", dest="rings", type=int, help="number of rings", default=1)
parser.add_argument("--force-device", type=str, choices=["cuda", "gpu", "cpu"])

args = parser.parse_args()
args.strategy = "semlc"
args.init_std = None
args.init_gabor = False
args.init_pretrain = False
args.group = None
args.auto_group = False


# SET UP EXPERIMENT
ax_client = AxClient(
    random_seed=111,
    verbose_logging=False
)

ax_client.create_experiment(
    parameters=[{
        "type": "range",
        "value_type": "float",
        "name": "w1",
        "bounds": [1.0, 15.0],
    }, {
        "type": "range",
        "value_type": "float",
        "name": "w2",
        "bounds": [1.0, 15.0],
    }, {
        "name": "r",
        "value_type": "float",
        "type": "range",
        "bounds": [0.0, 2.0],
    }, {
        "name": "d",
        "value_type": "float",
        "type": "range",
        "bounds": [0.0, 0.15],
    },
    ],
    # parameter_constraints=["w2 - w1 >= 1"],
    minimize=False,
    objective_name=args.metric
)


# SEARCH PHASE
for i in range(args.iterations):
    parameters, trial_index = None, None
    if is_root:
        parameters, trial_index = ax_client.get_next_trial()
        print(f"Generated Trial {trial_index} with parameters {parameters} on root.")

    parameters = comm.bcast(parameters)
    if not is_root:
        parameters, trial_index = ax_client.attach_trial(parameters=parameters)
    else:
        print(f"Attached {comm.size - 1} trials with parameters {parameters}\n")

    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters, other_args=args.__dict__))

    # save every iteration
    ax_client.save_to_json_file(f"experiments/static/hpoptims/{args.network}_{args.metric}_{hpoptim_id}.json")

# FINALIZATION
if is_root:
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(ax_client.generation_strategy.trials_as_df)

    best_parameters, values = ax_client.get_best_parameters()
    means, covariances = values
    print(f"Best parameters: {best_parameters}")
    print(f"Best score: {means}")

    # create an Ax report
    with open('report.html', 'w') as outfile:
        outfile.write(render_report_elements(
            "example_report",
            html_elements=[
                plot_config_to_html(ax_client.get_contour_plot("w1", "w2")),
                plot_config_to_html(ax_client.get_contour_plot("w1", "r")),
                plot_config_to_html(ax_client.get_contour_plot("w1", "d")),
                plot_config_to_html(ax_client.get_contour_plot("w2", "r")),
                plot_config_to_html(ax_client.get_contour_plot("w2", "d")),
                plot_config_to_html(ax_client.get_contour_plot("r", "d")),
            ],
            header=False,
        ))