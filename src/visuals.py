from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_SPLIT_ORDER = ("holdout", "unknown", "all")
_SPLIT_PANEL_ORDER = ("all", "holdout", "unknown")
_CLASS_ORDER = ("1-100", "101-1000", "1001-10000", "10001-n", "punct", "all")
_QUALITY_METRICS = ("lAcc", "lAcc (norm)", "CER (total)", "CER (errors)")
_SUBSET_ORDER = ("test", "school", "poetic_18", "poetic_19", "poetic_20")


def _higher_is_better(metric: str) -> bool:
    return metric.startswith("lAcc")


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
        lacc_by_model = subset.set_index("model name")["lAcc"]
        bars = sorted(ax.patches, key=lambda bar: bar.get_x())
        for bar, tick in zip(bars, ax.get_xticklabels(), strict=True):
            model = tick.get_text()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{lacc_by_model[model]:.2%} lAcc",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for tick in ax.get_xticklabels():
            model = tick.get_text()
            if model in colors:
                tick.set_color(colors[model])
                tick.set_fontweight('bold')

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.margins(y=0.12)

    fig.suptitle("Throughput by model", y=1.02)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def build_quality_table(
    quality_df: pd.DataFrame,
    subset_name: str,
    metric: str,
    model_names: list[str],
) -> pd.DataFrame:
    sub = quality_df[
        (quality_df["subset name"] == subset_name)
        & (quality_df["model name"].isin(model_names))
    ]
    table = sub.pivot(index="model name", columns=["split", "class"], values=metric)
    table = table.reindex(model_names)
    columns = [(split, cls) for split in _SPLIT_ORDER for cls in _CLASS_ORDER]
    table = table.reindex(columns=columns)
    table.index.name = "model name"
    return table


def build_quality_table_for_split(
    quality_df: pd.DataFrame,
    subset_name: str,
    split: str,
    metric: str,
    model_names: list[str],
) -> pd.DataFrame:
    sub = quality_df[
        (quality_df["subset name"] == subset_name)
        & (quality_df["split"] == split)
        & (quality_df["model name"].isin(model_names))
    ]
    table = sub.pivot(index="model name", columns="class", values=metric)
    table = table.reindex(model_names)
    table = table.reindex(columns=_CLASS_ORDER)
    table.index.name = "model name"
    return table


def style_quality_table(table: pd.DataFrame, metric: str) -> pd.io.formats.style.Styler:
    higher = _higher_is_better(metric)

    def bold_best(col: pd.Series) -> list[str]:
        target = col.max() if higher else col.min()
        return ["font-weight: bold" if np.isclose(v, target) else "" for v in col]

    return table.style.apply(bold_best, axis=0)


def quality_tables(
    quality_df: pd.DataFrame,
    model_names: list[str],
    *,
    metrics: tuple[str, ...] = _QUALITY_METRICS,
) -> dict[tuple[str, str], pd.io.formats.style.Styler]:
    """One styled table per (subset name, metric) pair."""
    subsets = [s for s in _SUBSET_ORDER if s in quality_df["subset name"].unique()]
    out: dict[tuple[str, str], pd.io.formats.style.Styler] = {}
    for subset_name in subsets:
        for metric in metrics:
            table = build_quality_table(quality_df, subset_name, metric, model_names)
            out[(subset_name, metric)] = style_quality_table(table, metric)
    return out


def _format_quality_value(value: float, metric: str) -> str:
    if metric.startswith("lAcc"):
        return f"{value:.2%}"
    return f"{value:.4f}"


def _best_value_mask(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    higher = _higher_is_better(metric)
    if higher:
        return table.apply(lambda col: np.isclose(col, col.max()), axis=0)
    return table.apply(lambda col: np.isclose(col, col.min()), axis=0)


def save_quality_table(
    quality_df: pd.DataFrame,
    subset_name: str,
    metric: str,
    model_names: list[str],
    *,
    save_path: Path | str,
) -> None:
    """Render quality tables (one per split) stacked vertically: all, holdout, unknown."""
    split_tables = {
        split: build_quality_table_for_split(
            quality_df, subset_name, split, metric, model_names
        )
        for split in _SPLIT_PANEL_ORDER
    }

    nrows = len(model_names)
    ncols = len(_CLASS_ORDER)
    panel_h = nrows * 0.42 + 0.9
    fig_w = max(10.0, ncols * 1.05)
    fig_h = panel_h * len(_SPLIT_PANEL_ORDER) + 1.2

    fig, axes = plt.subplots(
        len(_SPLIT_PANEL_ORDER),
        1,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.suptitle(f"{subset_name} — {metric}", fontsize=14, y=0.98)

    for ax, split in zip(axes.flat, _SPLIT_PANEL_ORDER, strict=True):
        table = split_tables[split]
        cell_text = table.map(lambda v: _format_quality_value(v, metric))
        best = _best_value_mask(table, metric)

        ax.axis("off")
        ax.set_title(split, fontsize=11, loc="center", pad=6)

        mpl_table = ax.table(
            cellText=cell_text.to_numpy(),
            rowLabels=table.index,
            colLabels=list(_CLASS_ORDER),
            loc="center",
            cellLoc="center",
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(8)
        mpl_table.scale(1.0, 1.3)

        for i in range(nrows):
            for j in range(ncols):
                cell = mpl_table[(i + 1, j)]
                if best.iloc[i, j]:
                    cell.set_text_props(fontweight="bold")

        for (row, col), cell in mpl_table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_text_props(fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


__all__ = (
    "plot_throughput",
    "build_quality_table",
    "build_quality_table_for_split",
    "style_quality_table",
    "quality_tables",
    "save_quality_table",
)
