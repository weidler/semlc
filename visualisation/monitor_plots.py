from copy import copy
from typing import List, Union, Dict

import numpy
from bokeh import plotting, embed, colors
from bokeh.models import HoverTool, ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
from scipy import stats

from config import CONFIG
from core.statistics import confidence_around_mean

palette = [colors.RGB(*[int(c * 255) for c in color]) for color in CONFIG.COLORMAP]
darker_palette = [c.darken(0.3) for c in palette]

tooltip_css = """
tooltip {
    background-color: #212121;
    color: white;
    padding: 5px;
    border-radius: 10px;
    margin-left: 10px;
}
"""

plot_styling = dict(
    plot_height=500,
    sizing_mode="stretch_width",
    toolbar_location=None,
    active_drag=None,
)


def style_plot(p):
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.outline_line_color = None

    p.axis.axis_label_text_font = "times"
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.axis_label_text_font_style = "bold"

    p.legend.label_text_font = "times"
    p.legend.label_text_font_size = "12pt"
    p.legend.label_text_font_style = "normal"

    p.title.align = "center"
    p.title.text_font_size = "14pt"
    p.title.text_font = "Fira Sans"

    p.y_range.start = 0


def render_progress_line_plot(epochs: list, measurements: Union[Dict[str, List[float]], Dict[str, List[List[float]]]],
                              metric="", title=""):
    lower_bounds, upper_bounds = {}, {}

    for key, accuracies in measurements.items():

        if isinstance(accuracies[0], list) and len(accuracies) == 1:
            accuracies = accuracies[0]

        # potentially combine multiple trainings into statistic
        lower_bounds[key], upper_bounds[key] = None, None
        if isinstance(accuracies[0], list):
            minimum_train_length = min(map(len, accuracies))
            accuracies = list(map(lambda x: x[:minimum_train_length], accuracies))
            epochs = epochs[:minimum_train_length]

            all_accuracies = copy(accuracies)
            accuracies = numpy.mean(all_accuracies, axis=0)
            accuracy_variances = numpy.std(all_accuracies, axis=0)

            lower_bounds[key], upper_bounds[key] = stats.norm.interval(0.99, loc=accuracies, scale=accuracy_variances)

            accuracies = accuracies.tolist()

        measurements[key] = accuracies

    source = ColumnDataSource(data=dict(
        epoch=epochs,
        **{k.replace('-', '_'): v for k, v in measurements.items()}
    ))

    tooltips = [("Epoch", "@epoch")]
    for group in measurements.keys():
        tooltips.append((f"{group}", f"@{group.replace('-', '_')}"))

    hovers = HoverTool(
        tooltips=tooltips,
        mode='vline'
    )

    y_range = dict()
    if metric.lower() == "accuracy":
        y_range = dict(y_range=(0, 100))

    fig = plotting.figure(x_axis_label="Epoch",
                          y_axis_label=metric,
                          **y_range,
                          x_range=(0, max(epochs)),
                          tools=[hovers],
                          title=title,
                          **plot_styling)

    for i, group in enumerate(measurements.keys()):
        val_acc_color = colors.RGB(*[int(c * 255) for c in CONFIG.COLORMAP[i]])
        if lower_bounds[group] is not None:
            fig.varea(epochs, lower_bounds[group], upper_bounds[group], color=val_acc_color.lighten(0.3),
                      fill_alpha=0.5)

    for i, group in enumerate(measurements.keys()):
        val_acc_color = colors.RGB(*[int(c * 255) for c in CONFIG.COLORMAP[i]])
        fig.line("epoch", group.replace('-', '_'),
                 legend_label=f"{group}",
                 color=val_acc_color,
                 line_width=2,
                 source=source)

    fig.legend.location = "bottom_right"
    style_plot(fig)

    return embed.components(fig)


def render_test_accuracy_plot(test_accuracies: Dict[str, List[Dict[str, Dict[str, List[float]]]]], metric="", title="",
                              error_rate=True):
    if len(test_accuracies) == 0:
        return None

    groups = list(test_accuracies)
    test_settings = list(test_accuracies[list(test_accuracies)[0]][0].keys())
    x = [(d, g) for d in test_settings for g in groups]
    y = []

    for dataset in test_settings:
        for group, accuracies in test_accuracies.items():
            # ci_lower_bounds[group], ci_upper_bounds[group] = {}, {}
            mean = confidence_around_mean(list(map(lambda d: d[dataset]["total"], test_accuracies[group])))[0]
            if error_rate:
                mean = 100 - mean
            y.append(mean)

    fig = plotting.figure(x_range=FactorRange(*x),
                          # x_axis_label="Transformation",
                          y_axis_label=metric,
                          title=title,
                          **plot_styling)

    source = ColumnDataSource(data=dict(x=x, y=y))
    fig.vbar(x="x", top="y", width=0.9, source=source,
             line_color=factor_cmap("x", palette=darker_palette, factors=groups, start=1, end=2),
             line_width=1,
             fill_color=factor_cmap("x", palette=palette, factors=groups, start=1, end=2))

    fig.xaxis.major_label_orientation = 1
    fig.add_tools(HoverTool(tooltips=[(metric, "@y")]))

    style_plot(fig)

    return embed.components(fig)


if __name__ == "__main__":
    import os
    import json

    exp_id = 1611533601916437
    path = f"../{CONFIG.MODEL_DIR}/{exp_id}"

    with open(os.path.join(path, "evaluation.json"), "r") as f:
        evaluation = json.load(f)

    accuracies = render_test_accuracy_plot(
        test_accuracies={f"{exp_id}": evaluation},
        title="Test Accuracy", metric="Accuracy"
    )
