from argparse import ArgumentParser, Namespace
from pathlib import Path

from src import (
    benchmark_lemmatization_quality,
    benchmark_throughput,
    get_sample_from_row_original,
)


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Run lemmatization quality and throughput benchmarks on CSV datasets."
    )
    parser.add_argument(
        "--quality-csvs",
        nargs="+",
        metavar="PATH",
        type=Path,
        default=[],
        help="One or more CSV paths for lemmatization quality benchmarking.",
    )
    parser.add_argument(
        "--speed-csvs",
        nargs="+",
        metavar="PATH",
        type=Path,
        default=[],
        help="One or more CSV paths for lemmatization speed benchmarking.",
    )
    return parser.parse_args(argv)


def _validate_csv_paths(paths: list[Path], arg_name: str) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{arg_name}: CSV not found: {path}")


def run_bench(argv: list[str] | None = None) -> Namespace:
    args = parse_args(argv)
    _validate_csv_paths(args.quality_csvs, "--quality-csvs")
    _validate_csv_paths(args.speed_csvs, "--speed-csvs")
    return args
