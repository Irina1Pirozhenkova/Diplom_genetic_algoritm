from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker


DEFAULT_SERIES_COLOR = "black"
LEGEND_FONT_SIZE = 8
LEGEND_LINE_SEP = 2


def render_curve(
    series: list[float] | list[tuple[str, list[float], str]],
    output_path: str | Path,
    title: str | None = None,
    marker_x: int | None = None,
    marker_y: float | None = None,
    marker_label: str | None = None,
    shock_axis_label: str | None = None,
    legend_lines: list[tuple[str, str]] | None = None,
    y_min: float | None = None,
    y_label: str | None = None,
) -> None:
    # The renderer accepts either one curve or several named/colorized series for comparison plots.
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if series and isinstance(series[0], tuple):
        series_specs = [(str(label), np.asarray(values, dtype=np.float64), color) for label, values, color in series]
    else:
        series_specs = [("", np.asarray(series, dtype=np.float64), DEFAULT_SERIES_COLOR)]

    x_values = np.arange(series_specs[0][1].size)
    all_values = np.concatenate([values for _, values, _ in series_specs])

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, values, color in series_specs:
        ax.plot(x_values, values, color=color, linewidth=1.4)

    ax.grid(True, color="#cfcfcf", linewidth=0.6, alpha=0.9)
    ax.set_xlabel("\u041f\u043e\u043a\u043e\u043b\u0435\u043d\u0438\u0435")
    ax.set_ylabel(y_label or "\u0421\u0443\u043c\u043c\u0430\u0440\u043d\u0430\u044f \u043f\u0430\u0441\u0441\u0438\u043e\u043d\u0430\u0440\u043d\u043e\u0441\u0442\u044c")
    if title:
        ax.set_title(title)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    if marker_x is not None and marker_y is not None and marker_label:
        # Shock annotation stays on the curve, while the generation number is pushed to the x-axis tick.
        x_span = max(len(x_values) * 0.08, 30.0)
        y_span = max(float(all_values.max()) * 0.10, 2.4)
        right_limit = float(x_values[-1]) if len(x_values) > 1 else float(marker_x)
        if marker_x > right_limit * 0.78:
            text_x = max(float(marker_x) - x_span, 6.0)
            horizontal = "right"
        else:
            text_x = min(float(marker_x) + x_span * 0.55, right_limit * 0.95 + 1.0)
            horizontal = "left"
        text_y = min(float(marker_y) + y_span, float(all_values.max()) * 0.94 + y_span * 0.1)
        bottom_y = ax.get_ylim()[0]
        ax.scatter([marker_x], [marker_y], color="#d62728", s=28, zorder=5)
        ax.vlines(
            marker_x,
            bottom_y,
            marker_y,
            colors="#d62728",
            linestyles="--",
            linewidth=0.95,
            zorder=4,
        )
        ax.annotate(
            marker_label,
            xy=(marker_x, marker_y),
            xytext=(text_x, text_y),
            color="#d62728",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "#d62728", "linewidth": 0.9},
            ha=horizontal,
            va="bottom",
        )
        if shock_axis_label:
            # We inject the shock generation into the normal tick set and recolor only that one label.
            ticks = [float(tick) for tick in ax.get_xticks()]
            ticks.append(float(marker_x))
            ticks = sorted(set(round(tick, 6) for tick in ticks if 0.0 <= tick <= float(x_values[-1])))
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(round(tick))) for tick in ticks])
            for tick_label, tick_value in zip(ax.get_xticklabels(), ticks, strict=False):
                if int(round(tick_value)) == int(marker_x):
                    tick_label.set_color("#d62728")

    if legend_lines:
        # The parameter panel is drawn inside the plotting area to behave like a compact experiment legend.
        text_items = [
            TextArea(text, textprops={"color": color, "fontsize": LEGEND_FONT_SIZE})
            for text, color in legend_lines
        ]
        legend_box = VPacker(children=text_items, align="left", pad=0, sep=LEGEND_LINE_SEP)
        anchored_box = AnchoredOffsetbox(
            loc="upper right",
            child=legend_box,
            pad=0.35,
            frameon=True,
            bbox_to_anchor=(0.985, 0.985),
            bbox_transform=ax.transAxes,
            borderpad=0.45,
        )
        anchored_box.patch.set_facecolor("white")
        anchored_box.patch.set_alpha(0.86)
        anchored_box.patch.set_edgecolor("#8a8a8a")
        anchored_box.patch.set_linewidth(0.8)
        ax.add_artist(anchored_box)

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
