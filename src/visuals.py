from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _build_palette(model_order: list[str], colors: dict[str, str]) -> dict[str, str]:
    default_colors = sns.color_palette("cool_r", n_colors=len(model_order))
    palette = dict(zip(model_order, default_colors, strict=True))
    palette.update({model: color for model, color in colors.items() if model in palette})
    return palette


def plot_throughput(
    throughput_df: pd.DataFrame,
    model_names: list[str],
    colors: dict[str, str] | None = None,
    *,
    metric: str = "LPS",
    figsize: tuple[float, float] = (14, 10),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """
    Draw four barplots for throughput across dtype and caching permutations.

    One subplot per combination of fp32/fp16 and caching enabled/disabled.
    Bars are sorted by speed within each subplot; y-axis scales independently.
    """
    if colors is None:
        colors = {}

    permutations = (
        ("fp32", False, "FP32, no caching"),
        ("fp32", True, "FP32, caching"),
        ("fp16", False, "FP16, no caching"),
        ("fp16", True, "FP16, caching"),
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for ax, (dtype, caching, title) in zip(axes_flat, permutations, strict=True):
        subset = throughput_df[
            (throughput_df["dtype"] == dtype)
            & (throughput_df["caching"] == caching)
            & (throughput_df["model name"].isin(model_names))
        ]
        order = (
            subset.sort_values(metric, ascending=False)["model name"]
            .tolist()
        )
        palette = _build_palette(order, colors)

        sns.barplot(
            data=subset,
            x="model name",
            y=metric,
            order=order,
            hue="model name",
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.margins(y=0.08)

    fig.suptitle("Throughput by model", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


__all__ = ("plot_throughput",)
