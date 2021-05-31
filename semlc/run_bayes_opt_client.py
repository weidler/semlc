import argparse
from typing import Any, Dict

import numpy
import pandas as pd
from ax.plot.render import plot_config_to_html
from ax.service.ax_client import AxClient
from ax.utils.report.render import render_report_elements
from mpi4py import MPI
from scipy import stats

comm = MPI.COMM_WORLD
rank = comm.rank
is_root = rank == 0


def train_evaluate(p: Dict[str, Any]) -> float:
    w1 = p["w1"] + numpy.random.randn() * 0.1
    w2 = p["w2"] + numpy.random.randn() * 0.1
    r = p["r"] + numpy.random.randn() * 0.1
    d = p["d"] + numpy.random.randn() * 0.1
    # w1 = p["w1"]
    # w2 = p["w2"]

    return (w1 + 2 * w2 - 7) ** 2 + (2 * w1 + w2 - 5) ** 2 * (r * d * - 2) * (- r / d) ** 3


def evaluate(p: Dict[str, Any]) -> Dict:
    process_acc = train_evaluate(p)
    gathered_means = comm.allgather(process_acc)

    if len(gathered_means) > 1:
        mean, sem = numpy.mean(gathered_means), stats.sem(gathered_means)
    else:
        mean, sem = gathered_means[0], 0.0

    return {"accuracy": (mean, sem)}


# PARSE CLIENT PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--iterations", type=int, default=10, help="Number of iterations the BO evaluates.")
parser.add_argument("--initial", type=int, default=5, help="Number of initial configs produced to setup BO.")

args = parser.parse_args()

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
        "bounds": [0.0, 10.0],
    }, {
        "type": "range",
        "value_type": "float",
        "name": "w2",
        "bounds": [0.0, 10.0],
    }, {
        "name": "r",
        "value_type": "float",
        "type": "range",
        "bounds": [0.0, 2.0],
    }, {
        "name": "d",
        "value_type": "float",
        "type": "range",
        "bounds": [0.0, 0.2],
    },
    ],
    # parameter_constraints=["w2 - w1 >= 1"],
    minimize=False,
    objective_name="accuracy"
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
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

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
