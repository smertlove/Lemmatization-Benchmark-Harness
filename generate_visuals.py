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

_README_PATH = Path("README.md")
_BENCHMARKS_START = "<!-- AUTO-GENERATED-BENCHMARKS:START -->"
_BENCHMARKS_END = "<!-- AUTO-GENERATED-BENCHMARKS:END -->"


def _quality_md_for_readme(path: Path) -> str:
    """Drop redundant top-level # title; keep HTML tables for GitHub rendering."""
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def update_root_readme(subsets: list[str]) -> None:
    """Refresh the auto-generated benchmark block in README.md."""
    quality_sections: list[str] = []
    for subset_name in subsets:
        md_path = QUALITY_OUT / f"{subset_name}.md"
        if not md_path.is_file():
            continue
        quality_sections.append(f"<h2> {subset_name}\n\n{_quality_md_for_readme(md_path)} </h2> <hr>")

    generated = f"""{_BENCHMARKS_START}

## Benchmark results

### Throughput

![Throughput by model, fp32/fp16 and caching](results/throughput_bars.png)

### Quality

{chr(10).join(quality_sections)}

{_BENCHMARKS_END}
"""
    readme = _README_PATH.read_text(encoding="utf-8")
    if _BENCHMARKS_START in readme and _BENCHMARKS_END in readme:
        before, rest = readme.split(_BENCHMARKS_START, 1)
        _mid, after = rest.split(_BENCHMARKS_END, 1)
        readme = before.rstrip() + "\n\n" + generated.rstrip() + "\n" + after.lstrip()
    else:
        readme = readme.rstrip() + "\n\n" + generated
    _README_PATH.write_text(readme, encoding="utf-8")


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

    update_root_readme(subsets)


if __name__ == "__main__":
    main()
