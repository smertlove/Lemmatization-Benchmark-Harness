"""Generate static benchmark artifacts from results/*.csv (throughput plot + quality tables)."""

from pathlib import Path

import pandas as pd

from src.visuals import (
    _SUBSET_ORDER,
    plot_throughput,
    write_quality_subset,
)

RESULTS = Path("results")
QUALITY_OUT = RESULTS / "quality"

THROUGHPUT_PATH = RESULTS / "throughput.csv"
QUALITY_PATH = RESULTS / "quality.csv"

# Throughput bar chart: highlight this model on the x-axis
HIGHLIGHT_COLORS = {"BART_4-4-404_66m": "red"}


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
    readme_parts = ["# Quality results\n"]
    for subset_name in subsets:
        md_path = QUALITY_OUT / f"{subset_name}.md"
        tex_path = QUALITY_OUT / f"{subset_name}.tex"
        write_quality_subset(
            quality_df,
            subset_name,
            model_names,
            md_path=md_path,
            tex_path=tex_path,
        )
        readme_parts.append(f"- [{subset_name}]({subset_name}.md)\n")

    (QUALITY_OUT / "README.md").write_text("".join(readme_parts), encoding="utf-8")


if __name__ == "__main__":
    main()
