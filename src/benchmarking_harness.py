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
    targets = df["lemma"]
    total = len(targets)

    timer = ThroughputTimer()
    calculator = Calculator()

    timer.start()
    preds = predict_fn(inpts)
    timer.stop(total)

    return {
        "lps": timer.lps,
        "lAcc": calculator.lAcc(targets, preds)
    }

