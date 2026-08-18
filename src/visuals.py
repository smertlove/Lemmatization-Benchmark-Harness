from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


_CLASS_ORDER = ("all", "1-100", "101-1000", "1001-10000", "10001-n", "punct")
_QUALITY_METRICS = ("lAcc", "lAcc (norm)", "CER (total)", "CER (errors)")
_SPLIT_PANEL_ORDER = ("all", "holdout", "unknown")
_SUBSET_ORDER = ("test", "school", "poetic_18", "poetic_19", "poetic_20")


def build_quality_table(
    quality_df: pd.DataFrame,
    subset_name: str,
    split: str,
    model_names: list[str],
) -> pd.DataFrame:
    """One split: rows are models, columns are (class, metric)."""
    sub = quality_df[
        (quality_df["subset name"] == subset_name)
        & (quality_df["split"] == split)
        & (quality_df["model name"].isin(model_names))
    ]
    blocks: list[pd.DataFrame] = []
    for cls in _CLASS_ORDER:
        block = (
            sub.loc[sub["class"] == cls]
            .set_index("model name")[list(_QUALITY_METRICS)]
            .reindex(model_names)
        )
        block.columns = pd.MultiIndex.from_product([[cls], list(_QUALITY_METRICS)])
        blocks.append(block)
    out = pd.concat(blocks, axis=1)
    out.index.name = "model name"
    return out


def _is_best_in_column(series: pd.Series, model: str, metric: str) -> bool:
    higher = _higher_is_better(metric)
    target = series.max() if higher else series.min()
    return bool(np.isclose(series.loc[model], target))


def _metric_label_md(metric: str) -> str:
    return {
        "lAcc": "lAcc",
        "lAcc (norm)": "lAcc_n",
        "CER (total)": "CER",
        "CER (errors)": "CER_e",
    }[metric]


def _metric_label_tex(metric: str) -> str:
    return {
        "lAcc": "lAcc",
        "lAcc (norm)": "lAcc$_n$",
        "CER (total)": "CER",
        "CER (errors)": "CER$_e$",
    }[metric]


def _tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
    )


def _split_table_html(data: pd.DataFrame, split: str) -> str:
    n_metrics = len(_QUALITY_METRICS)
    rows: list[str] = [f"<h3>{split}</h3>", "<table>", "<thead>"]
    rows.append('<tr><th rowspan="2">model</th>')
    for cls in _CLASS_ORDER:
        rows.append(f'<th colspan="{n_metrics}">{cls}</th>')
    rows.append("</tr><tr>")
    for _cls in _CLASS_ORDER:
        for metric in _QUALITY_METRICS:
            rows.append(f"<th>{_metric_label_md(metric)}</th>")
    rows.append("</tr></thead><tbody>")
    for model in data.index:
        rows.append(f"<tr><td>{model}</td>")
        for col in data.columns:
            val = data.loc[model, col]
            metric = col[1]
            if _is_best_in_column(data[col], model, metric):
                rows.append(f"<td><strong>{val:.2f}</strong></td>")
            else:
                rows.append(f"<td>{val:.2f}</td>")
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _split_table_tex(data: pd.DataFrame, split: str, subset_name: str) -> str:
    n_metrics = len(_QUALITY_METRICS)
    col_spec = "l|" + "|".join(["c" * n_metrics] * len(_CLASS_ORDER)) + "|"
    lines = [
        f"% {subset_name} — {split}",
        f"\\begin{{table}}[ht]",
        f"\\caption{{{_tex_escape(subset_name)} — {split}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]
    class_row = ["\\multicolumn{1}{c|}{}"]
    for cls in _CLASS_ORDER:
        class_row.append(f"\\multicolumn{{{n_metrics}}}{{c|}}{{{cls}}}")
    lines.append(" & ".join(class_row) + " \\\\")
    metric_row = ["model"]
    for _cls in _CLASS_ORDER:
        for metric in _QUALITY_METRICS:
            metric_row.append(_metric_label_tex(metric))
    lines.append(" & ".join(metric_row) + " \\\\")
    lines.append("\\hline")
    for model in data.index:
        cells = [_tex_escape(str(model))]
        for col in data.columns:
            val = data.loc[model, col]
            metric = col[1]
            cell = f"{val:.2f}"
            if _is_best_in_column(data[col], model, metric):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def write_quality_subset(
    quality_df: pd.DataFrame,
    subset_name: str,
    model_names: list[str],
    *,
    md_path: Path | str,
    tex_path: Path | str | None = None,
) -> None:
    """Write one subset as Markdown (HTML tables) and optional LaTeX."""
    md_parts = [f"# {subset_name}\n"]
    tex_parts = []
    for split in _SPLIT_PANEL_ORDER:
        data = build_quality_table(quality_df, subset_name, split, model_names)
        md_parts.append(_split_table_html(data, split))
        md_parts.append("")
        if tex_path is not None:
            tex_parts.append(_split_table_tex(data, split, subset_name))
    Path(md_path).write_text("\n".join(md_parts), encoding="utf-8")
    if tex_path is not None:
        Path(tex_path).write_text("\n".join(tex_parts), encoding="utf-8")


__all__ = (
    "plot_throughput",
    "build_quality_table",
    "write_quality_subset",
    "_SUBSET_ORDER",
)
