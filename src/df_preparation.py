import pandas as pd


def _get_freq_class(rank: str):
    """
    Определяет частотный класс по частотному рангу
    """

    if rank == "other":
        return "10001-n"

    if rank == "punct":
        return "punct"

    rank_int = int(rank)

    if rank_int < 101:
        return "1-100"

    if rank_int < 1001:
        return "101-1000"

    if rank_int < 10001:
        return "1001-10000"

    return "10001-n"


def add_freq_class(df: pd.DataFrame):
    """
    Добавляет колонку freq_class в датафрейм
    """

    df["freq_class"] = df["freq_rank"].map(_get_freq_class)

    return df


def get_sample_from_row_original(row):
    """
    Сампл для оригинальной модели.
    Возвращает входную строку.
    """
    form = row["form"]
    pos = row["pos"]
    feats = row["feats"]

    sample = " ".join(filter(lambda elem: pd.notna(elem), [form, pos, feats]))

    return sample


def filter_irrelevant(df):
    """
    Убирает из датафрема строки по которым мы не хотим считать статистики.
    """

    df_filtered = df[
        ~df["feats"].str.contains("Typo", na=False)
        & ~df["feats"].str.contains("Abbr", na=False)
        & ~df["feats"].str.contains("Anom", na=False)
        & ~df["feats"].str.contains("SYM", na=False)
    ]
    return df_filtered


__all__ = (
    "add_freq_class",
    "get_sample_from_row_original",
    "filter_irrelevant",
)
