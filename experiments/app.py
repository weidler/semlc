import ast
import math
import os
import re
import shutil
import statistics
from collections import Counter

import flask
import numpy
import simplejson as json
from flask import request
from flask_jsglue import JSGlue
from simplejson import JSONDecodeError

from config import CONFIG
from core.statistics import best_val_acc, best_loss, best_val_acc_epoch, _potentially_pad, best_test_acc, \
    conf_h_test_acc
from analysis.monitor_plots import render_progress_line_plot, render_test_accuracy_plot

app = flask.Flask(__name__, )
glue = JSGlue(app)


@app.route("/")
def overview():
    """Write Overview page."""
    exp_dir = CONFIG.MODEL_DIR
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    groups = Counter()
    experiments = {}
    info = {}
    for path in experiment_paths:
        eid_m = re.match("[0-9]+", str(path.split("/")[-1]))
        if eid_m:
            try:
                eid = eid_m.group(0)

                if os.path.isfile(os.path.join(path, "meta.json")) and os.path.isfile(os.path.join(path, "train.log")):
                    with open(os.path.join(path, "meta.json"), "r") as f:
                        meta = json.load(f)

                    with open(os.path.join(path, "train.log"), "r") as f:
                        progress = json.load(f)

                    experiments.update({
                        eid: dict(network=meta.get("network_type"),
                                  lateral_type=meta.get("lateral_type"),
                                  input_channels=meta.get("input_channels"),
                                  dataset=meta.get("dataset").get("name"),
                                  host=meta.get("host"),
                                  group=meta.get("group"),
                                  peak=round(max(progress.get("val_acc")), 2),
                                  epochs=max(progress.get("epoch")),
                                  ended_naturally=meta.get("ended_naturally"))
                    })

                    groups.update([meta.get("group")])
            except Exception as e:
                continue

        info.update(dict(
            groups=groups,
        ))

    return flask.render_template("overview.html", exps=experiments, info=info)


@app.route("/groups")
def groupview():
    """Write group view page."""
    exp_dir = CONFIG.MODEL_DIR
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    groups = {}
    for path in experiment_paths:
        eid_m = re.match("[0-9]+", str(path.split("/")[-1]))
        if eid_m:
            try:
                if os.path.isfile(os.path.join(path, "meta.json")) and os.path.isfile(os.path.join(path, "train.log")):
                    with open(os.path.join(path, "meta.json"), "r") as f:
                        meta = json.load(f)

                    with open(os.path.join(path, "train.log"), "r") as f:
                        progress = json.load(f)

                    # skip no progress experiments
                    if len(progress.get("epoch")) < 1:
                        continue

                    # initialize info for group if not yet recorded
                    if meta.get("group") not in groups:
                        groups.update({meta.get("group"): {
                            "count": 0,
                            "network_types": set(),
                            "datasets": set(),
                            "lateral_types": set(),
                            "epochs": []
                        }})

                    # add experiment info to group info
                    groups[meta.get("group")]["count"] += 1
                    groups[meta.get("group")]["network_types"].add(meta.get("network_type"))
                    groups[meta.get("group")]["lateral_types"].add(meta.get("lateral_type"))
                    groups[meta.get("group")]["datasets"].add(meta.get("dataset").get("name"))
                    groups[meta.get("group")]["epochs"].append(max(progress.get("epoch")))

            except:
                continue

    for group in groups.keys():
        groups[group]["epochs"] = round(statistics.mean(groups[group]["epochs"]), 2)

    return flask.render_template("groupview.html", groups=groups)


@app.route("/experiment/<int:exp_id>", methods=("POST", "GET"))
def show_experiment(exp_id):
    """Show experiment of given ID."""
    experiment_paths = sorted([int(p) for p in os.listdir(f"{CONFIG.MODEL_DIR}")])
    current_index = experiment_paths.index(exp_id)

    path = f"{CONFIG.MODEL_DIR}/{exp_id}"
    with open(os.path.join(path, "train.log"), "r") as f:
        progress = json.load(f)

    with open(os.path.join(path, "meta.json"), "r") as f:
        meta = json.load(f)

    info = dict(network=meta.get("network_type"),
                dataset=meta.get("dataset"),
                is_lateral=meta.get("is_lateral"),
                lateral_type=meta.get("lateral_type", "unknown"),
                input_channels=meta.get("input_channels"),
                host=meta.get("host"),
                group=meta.get("group"),
                epochs=max(progress["epoch"]) if "epoch" in meta else None,
                hps={
                    **(meta.get("lateral_layer") if meta.get("lateral_layer") is not None else {})
                },
                current_id=exp_id,
                next_id=experiment_paths[current_index + 1] if current_index != len(experiment_paths) - 1 else None,
                prev_id=experiment_paths[current_index - 1] if current_index != 0 else None,
                accuracy_progress_plot=render_progress_line_plot(epochs=progress["epoch"],
                                                                 measurements={f"{exp_id}": progress.get("val_acc")},
                                                                 title="Validation Accuracy", metric="Accuracy"),
                loss_progress_plot=render_progress_line_plot(epochs=progress["epoch"],
                                                             measurements={"Validation": progress.get("val_loss"),
                                                                           "Training": progress.get(
                                                                               "train_loss") if "train_loss" in progress else progress.get(
                                                                               "loss")},
                                                             title="Loss", metric="Cross Entropy Loss")
                )

    info["test_accuracies"] = []
    if os.path.isfile(os.path.join(path, "evaluation.json")):
        with open(os.path.join(path, "evaluation.json"), "r") as f:
            evaluation = json.load(f)

        info["evaluated_settings"] = evaluation.keys()
        info["test_accuracies"] = render_test_accuracy_plot(
            test_accuracies={f"{exp_id}": [evaluation]},
            title="Test Accuracy", metric="Accuracy"
        )

    return flask.render_template("experiment.html", info=info)


@app.route("/analyze/", methods=("POST", "GET"))
def analyze():
    """Show experiment of given ID."""
    exp_ids = request.args.getlist('id')
    progress_dicts = {}
    meta_dicts = {}
    evaluation_dicts = {}

    max_epoch = 0

    for id in exp_ids:
        path = f"{CONFIG.MODEL_DIR}/{id}"

        with open(os.path.join(path, "train.log"), "r") as f:
            progress_dicts[id] = json.load(f)
            max_epoch = max(max(progress_dicts[id]["epoch"]), max_epoch)

        with open(os.path.join(path, "meta.json"), "r") as f:
            meta_dicts[id] = json.load(f)

        if os.path.isfile(os.path.join(path, "evaluation.json")):
            with open(os.path.join(path, "evaluation.json"), "r") as f:
                evaluation_dicts[id] = json.load(f)

    info = dict(ids=exp_ids,
                network=list(meta_dicts.values())[0].get("network_type"),
                dataset=list(meta_dicts.values())[0].get("dataset"),
                is_lateral=list(meta_dicts.values())[0].get("is_lateral"),
                lateral_type=list(meta_dicts.values())[0].get("lateral_type", "unknown"),
                hps={},

                mean_best_val_acc=round(best_val_acc([v.get("val_acc") for v in progress_dicts.values()]), 5),
                mean_best_val_acc_epoch=round(best_val_acc_epoch([v.get("val_acc") for v in progress_dicts.values()]),
                                              2),
                mean_best_loss=round(best_loss(
                    [v.get("train_loss") if "train_loss" in v else v.get("loss") for v in progress_dicts.values()]), 5),
                val_accuracy=render_progress_line_plot(epochs=list(range(1, max_epoch + 1)),
                                                       measurements={"all samples": [v.get("val_acc") for v in
                                                                                     progress_dicts.values()]},
                                                       title="Validation Accuracy", metric="Accuracy"),
                loss_progress_plot=render_progress_line_plot(epochs=list(range(1, max_epoch + 1)),
                                                             measurements={"Validation": [v.get("val_loss") for v in
                                                                                          progress_dicts.values()],
                                                                           "Training": [
                                                                               v.get(
                                                                                   "train_loss") if "train_loss" in v else v.get(
                                                                                   "loss") for v in
                                                                               progress_dicts.values()]},
                                                             title="Loss", metric="Cross Entropy Loss"),

                mean_train_time=numpy.array(
                    [v.get("train_time") if "train_time" in v else math.inf for v in meta_dicts.values()]).mean().round(
                    2).item(),
                mean_epoch_time=numpy.array([v.get("time_per_epoch") if "time_per_epoch" in v else math.inf for v in
                                             meta_dicts.values()]).mean().round(2).item(),
                )

    if len(evaluation_dicts) == len(progress_dicts):
        info["evaluation"] = dict(
            mean_test_acc=round(numpy.array([d["default"]["total"] for d in evaluation_dicts.values()]).mean().item(),
                                5)
        )
    else:
        info["evaluation"] = False

    return flask.render_template("analysis.html", info=info)


@app.route("/compare/", methods=("POST", "GET"))
def compare():
    """Compare given groups."""
    group_names = request.args.getlist('name')

    progress_dicts = {g: {} for g in group_names}
    meta_dicts = {g: {} for g in group_names}
    evaluation_dicts = {}

    max_epoch = 0

    exp_dir = CONFIG.MODEL_DIR
    experiment_paths = [os.path.join(exp_dir, p) for p in os.listdir(exp_dir)]

    for path in experiment_paths:
        eid_m = re.match("[0-9]+", str(path.split("/")[-1]))
        if eid_m:
            try:
                with open(os.path.join(path, "meta.json"), "r") as f:
                    meta = json.load(f)
            except JSONDecodeError:
                continue

            group = meta.get("group")

            if group not in group_names:
                continue
            with open(os.path.join(path, "train.log"), "r") as f:
                progress = json.load(f)

                if not progress["epoch"]:
                    continue

                max_epoch = max(max(progress["epoch"]), max_epoch)

            meta_dicts[group][eid_m.group(0)] = meta
            progress_dicts[group][eid_m.group(0)] = progress

            if os.path.isfile(os.path.join(path, "evaluation.json")):
                if group not in evaluation_dicts:
                    evaluation_dicts[group] = {}

                with open(os.path.join(path, "evaluation.json"), "r") as f:
                    evaluation_dicts[group][eid_m.group(0)] = json.load(f)

    stats = {}
    for g in group_names:
        progress_values = progress_dicts[g].values() if g in progress_dicts else []
        evaluation_values = evaluation_dicts[g].values() if g in evaluation_dicts else []

        stats.update({g: dict(
            mean_best_val_acc=round(best_val_acc([e.get("val_acc") for e in progress_values]), 2),
            mean_best_val_acc_epoch=round(best_val_acc_epoch([e.get("val_acc") for e in progress_values]), 2),
            mean_best_val_loss=round(best_loss([e.get("val_loss") if "val_loss" in e else e.get("loss") for e in progress_values]), 2),
            mean_best_test_acc=round(best_test_acc([e.get("default").get("total") for e in evaluation_values]), 2),
            conf_h_test_acc=round(conf_h_test_acc([e.get("default").get("total") for e in evaluation_values]), 2),
        )})

        stats[g]["is_best"] = False
        stats[g]["is_worst"] = False

    all_mean_test_accuracies = numpy.sort(numpy.array([v["mean_best_test_acc"] for g, v in stats.items()]))
    all_mean_test_accuracies = all_mean_test_accuracies[~numpy.isnan(all_mean_test_accuracies)]

    for g in stats.keys():
        if stats[g]["mean_best_test_acc"] >= all_mean_test_accuracies[-1]:
            stats[g]["is_best"] = True
        if stats[g]["mean_best_test_acc"] <= all_mean_test_accuracies[0]:
            stats[g]["is_worst"] = True

    data = dict(
        groups=group_names,
        group_stats=stats,
        group_colors=CONFIG.COLORMAP_HEX[:len(group_names)],
        accuracy_progress_plot=render_progress_line_plot(epochs=list(range(1, max_epoch + 1)),
                                                         measurements={g: [exp.get("val_acc") for exp in
                                                                           progress_dicts[g].values()] for g in
                                                                       progress_dicts.keys()},
                                                         title="Validation Accuracy", metric="Accuracy"),
        loss_progress_plot=render_progress_line_plot(epochs=list(range(1, max_epoch + 1)),
                                                     measurements={g: [exp.get("val_loss") for exp in
                                                                       progress_dicts[g].values()] for g in
                                                                   progress_dicts.keys()},
                                                     title="Validation Loss", metric="Cross Entropy Loss"),
        test_accuracies=render_test_accuracy_plot(test_accuracies={g: list(evaluation_dicts[g].values()) for g in
                                                                   evaluation_dicts.keys()},
                                                  title="Error Rates", metric="Accuracy"),
    )

    return flask.render_template("comparison.html", info=data)


@app.route("/evaluate_experiments", methods=("POST", "GET"))
def evaluate_experiments():
    """Render plots for an experiment in D3."""
    if request.method == "POST":
        try:
            exp_ids = ast.literal_eval(request.json['ids'])

            for i, id in enumerate(exp_ids):
                print(f"({i}/{len(exp_ids)}): {id}")
                os.chdir(CONFIG.PROJECT_PATH)
                os.system(f"python evaluate.py {id}")

            return {"success": "doing"}

        except Exception as e:
            return {"error": str(e)}
    else:
        return {"success": "no post"}


@app.route("/render_plots/<int:exp_id>/", methods=("POST", "GET"))
def render_plots(exp_id):
    """Render plots for an experiment in D3."""
    if request.method == "POST":
        try:
            path = f"{CONFIG.MODEL_DIR}/{exp_id}"

            with open(os.path.join(path, "train.log"), "r") as f:
                progress = json.load(f)

            with open(os.path.join(path, "meta.json"), "r") as f:
                meta = json.load(f)

            if os.path.isfile(os.path.join(path, "evaluation.json")):
                with open(os.path.join(path, "evaluation.json"), "r") as f:
                    evaluation = json.load(f)
            else:
                evaluation = {}

            return dict(
                val_accuracy=render_progress_line_plot(progress["epoch"],
                                                       _potentially_pad(progress["val_acc"], pad_val=None).tolist()),
                test_accuracy={
                    setting: render_test_accuracy_plot(
                        results["total"],
                        results["balanced"],
                        results["categories"],
                        meta["dataset"]["classes"]
                    ) for setting, results in evaluation.items()
                }
            )

        except Exception as e:
            return {"error": str(e)}
    else:
        return {"success": "no post"}


@app.route("/_clear_all_empty")
def clear_all_empty():
    """Delete all experiments stored that have less than 2 episodes finished."""
    experiment_paths = [os.path.join(CONFIG.MODEL_DIR, p) for p in os.listdir(CONFIG.MODEL_DIR)]

    deleted = 0
    for path in experiment_paths:
        if re.match("[0-9]+", str(path.split("/")[-1])):
            if os.path.isfile(os.path.join(path, "train.log")):
                try:
                    with open(os.path.join(path, "train.log"), "r") as f:
                        progress = json.load(f)
                except Exception:
                    # delete corrupted
                    shutil.rmtree(path)
                    deleted += 1
                    continue

                if "epoch" in progress and len(progress["epoch"]) < 1:
                    shutil.rmtree(path)
                    deleted += 1

    return {"deleted": deleted}


@app.route("/_clear_all_short")
def clear_all_short():
    """Delete all experiments stored that have less than 10 cycles finished."""
    experiment_paths = [os.path.join(CONFIG.MODEL_DIR, p) for p in os.listdir(CONFIG.MODEL_DIR)]

    deleted = 0
    for path in experiment_paths:
        if re.match("[0-9]+", str(path.split("/")[-1])):
            if os.path.isfile(os.path.join(path, "train.log")):

                try:
                    with open(os.path.join(path, "train.log"), "r") as f:
                        progress = json.load(f)
                except Exception:
                    # delete corrupted
                    shutil.rmtree(path)
                    deleted += 1
                    continue

                if "epoch" in progress and len(progress["epoch"]) <= 20:
                    shutil.rmtree(path)
                    deleted += 1

    return {"deleted": deleted}


@app.route("/_clear_all_unfinished")
def clear_all_unfinished():
    """Delete all experiments that did not finish all epochs naturally."""
    experiment_paths = [os.path.join(CONFIG.MODEL_DIR, p) for p in os.listdir(CONFIG.MODEL_DIR)]

    deleted = 0
    for path in experiment_paths:
        if re.match("[0-9]+", str(path.split("/")[-1])):
            if os.path.isfile(os.path.join(path, "train.log")):

                try:
                    with open(os.path.join(path, "train.log"), "r") as f:
                        progress = json.load(f)

                    with open(os.path.join(path, "meta.json"), "r") as f:
                        meta = json.load(f)

                except Exception as e:
                    # delete corrupted
                    shutil.rmtree(path)
                    deleted += 1
                    continue

                if "ended_naturally" in meta and not meta["ended_naturally"]:
                    shutil.rmtree(path)
                    deleted += 1

    return {"deleted": deleted}


@app.route("/_clear_group/", methods=['GET', 'POST'])
def clear_group():
    """Delete all experiments of a group."""
    experiment_paths = [os.path.join(CONFIG.MODEL_DIR, p) for p in os.listdir(CONFIG.MODEL_DIR)]

    if request.method == 'POST':
        group = request.json['group']

        deleted = 0
        for path in experiment_paths:
            if re.match("[0-9]+", str(path.split("/")[-1])):
                if os.path.isfile(os.path.join(path, "train.log")):

                    try:
                        with open(os.path.join(path, "meta.json"), "r") as f:
                            meta = json.load(f)

                    except Exception as e:
                        # delete corrupted
                        shutil.rmtree(path)
                        deleted += 1
                        continue

                    if "group" in meta and meta["group"] == group:
                        shutil.rmtree(path)
                        deleted += 1

        return {"deleted": deleted}

    return {"deleted": 0}
