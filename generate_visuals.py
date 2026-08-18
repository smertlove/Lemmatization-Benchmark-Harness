from pathlib import Path

import pandas as pd

from src.visuals import (
    _QUALITY_METRICS,
    _SUBSET_ORDER,
    plot_throughput,
    save_quality_table,
)

RESULTS = Path("results")
QUALITY_OUT = RESULTS / "quality"

THROUGHPUT_PATH = RESULTS / "throughput.csv"
QUALITY_PATH = RESULTS / "quality.csv"

HIGHLIGHT_COLORS = {"BART_4-4-404_66m": "red"}


def _metric_slug(metric: str) -> str:
    return metric.replace(" ", "_").replace("(", "").replace(")", "")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    QUALITY_OUT.mkdir(parents=True, exist_ok=True)

    throughput_df = pd.read_csv(THROUGHPUT_PATH, sep="\t")
    quality_df = pd.read_csv(QUALITY_PATH, sep="\t")
    model_names = throughput_df["model name"].unique().tolist()

    plot_throughput(
        throughput_df,
        model_names,
        HIGHLIGHT_COLORS,
        save_path=RESULTS / "throughput_bars.png",
    )

    subsets = [s for s in _SUBSET_ORDER if s in quality_df["subset name"].unique()]
    for subset_name in subsets:
        for metric in _QUALITY_METRICS:
            save_quality_table(
                quality_df,
                subset_name,
                metric,
                model_names,
                save_path=QUALITY_OUT / f"{subset_name}_{_metric_slug(metric)}.png",
            )


if __name__ == "__main__":
    main()
