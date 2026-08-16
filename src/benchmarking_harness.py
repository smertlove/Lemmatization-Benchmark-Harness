from .df_preparation import filter_irrelevant, add_freq_class
from .metrics import Calculator, ThroughputTimer

import pandas as pd
from pathlib import Path


def _load_df(path: Path):
    assert path.exists()
    return pd.read_csv(path, sep="\t")


def _save_df(df: pd.DataFrame, name: Path):
    df.to_csv(name, sep="\t", index=None)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def _benchmark_throughput_single_df(predict_fn, clear_cache_fn, df: pd.DataFrame, get_sample_from_row):
    """
    Замеряет lps и lAcc.
    lps -- Lemmas per Second, кол-во лемм, которые мы генерируем за секунду.
    lAcc -- Lemmatization Accuracy, см. src/metrics.py

    lAcc мерим чтобы понять, что predict_fn действительно работает как надо и мы не получаем там мусорные предсказания из-за ошибки.
    """

    clear_cache_fn()

    inpts = df.apply(get_sample_from_row, axis=1).tolist()
    targets = df["lemma"].tolist()
    total = len(targets)

    timer = ThroughputTimer()
    calculator = Calculator()

    timer.start()
    preds = predict_fn(inpts)
    timer.stop(total)

    return {"LPS": timer.lps, "lAcc": calculator.lAcc(targets, preds)}


def benchmark_throughput(
    predict_fn,
    clear_cache_fn,
    model_name: str,
    get_sample_from_row,
    throughput_csvs_paths: list[Path],
    throughput_table_path: Path,
    dtype,
    caching,
):
    throughput_table = _load_df(throughput_table_path)
    assert model_name not in throughput_table["model name"].unique(), f"{model_name} already logged."

    for p in throughput_csvs_paths:
        print(f"Processing {p.name}...")
        subset_name = p.stem

        df = _load_df(p)
        cur_metrics = _benchmark_throughput_single_df(predict_fn, clear_cache_fn, df, get_sample_from_row)
        cur_metrics["dtype"] = dtype
        cur_metrics["caching"] = caching
        cur_metrics["subset name"] = subset_name
        cur_metrics["model name"] = model_name

        row = pd.DataFrame([cur_metrics])
        throughput_table = pd.concat([throughput_table, row], ignore_index=True)

    _save_df(throughput_table, throughput_table_path)


def _calc_metrics_by_freq_class(df: pd.DataFrame):

    freq_groups = {
        "1-100": df[df["freq_class"] == "1-100"],
        "101-1000": df[df["freq_class"] == "101-1000"],
        "1001-10000": df[df["freq_class"] == "1001-10000"],
        "10001-n": df[df["freq_class"] == "10001-n"],
        "punct": df[df["freq_class"] == "punct"],
        "all": df,
    }

    metrics = []

    for freq_class, group in freq_groups.items():
        errors = group[group["lAcc"] == 0.0]

        metrics.append(
            {
                "class": freq_class,
                "lAcc": _safe_mean(group["lAcc"]),
                "lAcc (norm)": _safe_mean(group["lAcc (norm)"]),
                "CER (total)": _safe_mean(group["CER"]),
                "CER (errors)": _safe_mean(errors["CER"]),
            }
        )

    return metrics

def _benchmark_lemmatization_quality_single_df(predict_fn, df: pd.DataFrame, get_sample_from_row):

    ## Фильтры и сплиты выносим сюда потому что некоторые датафреймы у нас очень большие и долго предобрабатываются 
    df = filter_irrelevant(df)
    df = add_freq_class(df)
    df["inpt"] = df.apply(get_sample_from_row, axis=1).tolist()

    ## По той же причине здесь греем кеш: без этого действия замер на только одном сабсете растягивается на несколько часов
    uniq_samples = df["inpt"].unique().tolist()
    predict_fn(uniq_samples)

    ## Т.к. кеш прогрет, вот эта операция уже должна быть примерно моментальной
    inpts = df["inpt"].tolist()
    df["pred"] = predict_fn(inpts)

    calculator = Calculator()

    ## Метрики для каждой пары тоже достаточно посчитать только один раз
    df["lAcc"] = df.apply(
        lambda row: calculator.lAcc(row["lemma"], row["pred"]), axis=1
    )

    df["lAcc (norm)"] = df.apply(
        lambda row: calculator.lAcc(row["lemma"], row["pred"], normalize=True), axis=1
    )

    df["CER"] = df.apply(lambda row: calculator.CER(row["lemma"], row["pred"]), axis=1)

    ## Мы не мерим здесь normalized CER потому что эта метрика примерно ни о чем нам не говорит.

    ## Теперь расчитаем метрики для всех категорий которые нам нужны

    all_metrics = []

    for cur_df, split_name in (
        (df[df["split"] == "holdout"], "holdout"),
        (df[df["split"] == "unknown"], "unknown"),
        (df, "all"),
    ):
        cur_metrics = _calc_metrics_by_freq_class(cur_df)
        for row in cur_metrics:
            row["split"] = split_name

        all_metrics.extend(cur_metrics)

    return all_metrics


def benchmark_lemmatization_quality(
    predict_fn,  # Note: это должен быть predict_fast или любая штука, активно привлекающая кеширование
    model_name: str,
    get_sample_from_row,
    quality_csvs_paths: list[Path],
    quality_table_path: Path,
):
    """
    для корректной работы аргумент predict_fn должен быть predict_fast или любой штукой, активно привлекающей кеширование
    """

    quality_table = _load_df(quality_table_path)
    assert model_name not in quality_table["model name"].unique(), f"{model_name} already logged."

    for p in quality_csvs_paths:

        print(f"Processing {p.name}...")
        subset_name = p.stem

        df = _load_df(p)
        cur_metrics = _benchmark_lemmatization_quality_single_df(predict_fn, df, get_sample_from_row)
        for row in cur_metrics:
            row["subset name"] = subset_name
            row["model name"] = model_name

        df = pd.DataFrame(cur_metrics)
        quality_table = pd.concat([quality_table, df], ignore_index=True)

    _save_df(quality_table, quality_table_path)
