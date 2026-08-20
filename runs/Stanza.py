import gc
import math
from pathlib import Path
import pandas as pd
import stanza
from stanza.models.common.doc import Document
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from src import (
    GenerativeModelWithCachingAndHeuristics,
    benchmark_lemmatization_quality,
    benchmark_throughput,
    sanity_check,
)

data_dir = Path("/mnt/data_storage/datasets/generative_lemmatization_datasets/csvs/bench")
models_dir = Path("/mnt/data_storage/models/generative_lemmatization")

quality_table = Path("./results/quality.csv")
throughput_table = Path("./results/throughput.csv")

throughput_csvs = [
    data_dir / "test.csv",
]

quality_csvs = [
    data_dir / "test.csv",
    data_dir / "school.csv",
    data_dir / "poetic_18.csv",
    data_dir / "poetic_19.csv",
    data_dir / "poetic_20.csv",
]

MODEL_NAME = "Stanza"
MODEL_ID = models_dir / MODEL_NAME
BATCH_SIZE = 512
CACHE_SIZE = 5_000_000


def get_sample_from_row_stanza(row):
    """
    Выход этой штуки должен быть hashable, поэтому приходится изголяться
    """
    form = row["form"] if pd.notna(row["form"]) else ""
    pos = row["pos"] if pd.notna(row["pos"]) else ""
    feats = row["feats"] if pd.notna(row["feats"]) else ""

    sample = f"{form}$${pos}$${feats}"

    return sample

def _get_sample_from_row_stanza(row_with_dollars):

    form, pos, feats = row_with_dollars.split("$$", maxsplit=2)
    if not form:
        form = "ERR_TOKEN"

    return [{'id': 1, 'text': form, 'upos': pos, "feats": feats}]


def sanity_check(gen_model) -> None:
    predict_fn = lambda texts: gen_model.predict_fast(texts, batch_size=2)

    cases = [
        "дырой NOUN Animacy:Inan Case:Ins Gender:Fem Number:Sing",
        "норой NOUN Animacy:Inan Case:Ins Gender:Fem Number:Sing",
    ]
    golds = ["дыра", "нора"]

    preds = predict_fn(cases)

    assert len(preds) == len(golds)
    for pred, gold in zip(preds, golds):
        assert pred == gold, f"{pred} != {gold}"

    assert len(gen_model._cache) == 2
    gen_model.clear_cache()
    assert len(gen_model._cache) == 0


if __name__ == "__main__":

    nlp = stanza.Pipeline('ru', processors='tokenize,lemma', use_gpu=True , tokenize_pretokenized=True, lemma_pretagged=True,)

    def lemmatize_in_chunks(words, chunk_size=512, nlp=nlp, verbose=True):
        lemmas = []

        if verbose:
            pbar: range | tqdm[int] = tqdm(
                range(0, len(words), chunk_size),
                total=math.ceil(len(words) / chunk_size),
            )
        else:
            pbar = range(0, len(words), chunk_size)

        for i in pbar:
            chunk = words[i : i + chunk_size]

            doc = Document([_get_sample_from_row_stanza(elem) for elem in chunk])

            parsed_doc = nlp(doc)

            for sent in parsed_doc.sentences:
                lemmas.append(sent.words[0].lemma)

        return lemmas

    ## ==== Sanity check ==== ##

    cases = [
        "дырой$$NOUN$$Animacy:Inan Case:Ins Gender:Fem Number:Sing",
        "норой$$NOUN$$Animacy:Inan Case:Ins Gender:Fem Number:Sing",
    ]
    
    golds = ["дыра", "нора"]

    preds = lemmatize_in_chunks(cases)

    assert len(preds) == len(golds)
    for pred, gold in zip(preds, golds):
        assert pred == gold, f"{pred} != {gold}"

    ## ==== fp32 ==== ##

    benchmark_throughput(
        lemmatize_in_chunks,
        lambda: ...,
        MODEL_NAME,
        get_sample_from_row_stanza,
        throughput_csvs,
        throughput_table,
        "fp32",
        False,
    )

    benchmark_lemmatization_quality(
        lemmatize_in_chunks,
        MODEL_NAME,
        get_sample_from_row_stanza,
        quality_csvs,
        quality_table,
    )


