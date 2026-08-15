from df_preparation import filter_irrelevant, add_freq_class
from metrics import Calculator, ThroughputTimer

import pandas as pd


def benchmark_throughput(predict_fn, df: pd.DataFrame, get_sample_from_row):
    """
    Замеряет lps и lAcc.
    lps -- Lemmas per Second, кол-во лемм, которые мы генерируем за секунду.
    lAcc -- Lemmatization Accuracy, см. src/metrics.py

    lAcc мерим чтобы понять, что predict_fn действительно работает как надо и мы не получаем там мусорные предсказания из-за ошибки.
    """
    inpts = df.apply(get_sample_from_row, axis=1).tolist()
    targets = df["lemma"].tolist()
    total = len(targets)

    timer = ThroughputTimer()
    calculator = Calculator()

    timer.start()
    preds = predict_fn(inpts)
    timer.stop(total)

    return {"lps": timer.lps, "lAcc": calculator.lAcc(targets, preds)}


def benchmark_lemmatization_quality(predict_fn, df: pd.DataFrame, get_sample_from_row):
    df = filter_irrelevant(df)
    df = add_freq_class(df)

    inpts = df.apply(get_sample_from_row, axis=1).tolist()
    df["pred"] = predict_fn(inpts)

    calculator = Calculator()

    df["lAcc"] = df.apply(
        lambda row: calculator.lAcc(row["lemma"], row["pred"]), axis=1
    )

    df["lAcc (norm)"] = df.apply(
        lambda row: calculator.lAcc(row["lemma"], row["pred"], normalize=True), axis=1
    )

    df["CER"] = df.apply(lambda row: calculator.CER(row["lemma"], row["pred"]), axis=1)

    ## Мы не мерим здесь CER потому что эта метрика примерно ни о чем нам не говорит.

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
                "freq_class": freq_class,
                "lAcc": group["lAcc"].mean(),
                "lAcc (norm)": group["lAcc (norm)"].mean(),
                "CER (total)": group["CER"].mean(),
                "CER (errors)": errors["CER"].mean() if not errors.empty else 0.0,
            }
        )

    return pd.DataFrame(metrics)
