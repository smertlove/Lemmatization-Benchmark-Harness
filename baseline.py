from src import (
    benchmark_lemmatization_quality,
    benchmark_throughput,
    GenerativeModelWithCachingAndHeuristics,
    get_sample_from_row_original
)
from pathlib import Path
from transformers import BartForConditionalGeneration, AutoTokenizer


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

NODEL_NAME = "baseline"
MODEL_ID = models_dir / NODEL_NAME
BATCH_SIZE = 512

## ==== Sanity check ==== ##

model = BartForConditionalGeneration.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

gen_model = GenerativeModelWithCachingAndHeuristics(model, tokenizer, cache_size=5000000)

predict_fn = lambda texts: gen_model.predict_fast(texts, batch_size=BATCH_SIZE)
clear_cache_fn = gen_model.clear_cache

cases = [
    "дырой NOUN Animacy:Inan Case:Ins Gender:Fem Number:Sing",
    "норой NOUN Animacy:Inan Case:Ins Gender:Fem Number:Sing",
]

golds = ["дыра", "нора",]

preds = predict_fn(cases)

assert len(preds) == len(golds)
for pred, gold in zip(preds, golds):
    assert pred == gold, f"{pred} != {gold}"

assert len(gen_model._cache) == 2
clear_cache_fn()
assert len(gen_model._cache) == 0

## ==== Benchmark throughput ==== ##
## fp32; no caching

predict_fn = lambda texts: gen_model.predict(texts, batch_size=BATCH_SIZE)
clear_cache_fn = gen_model.clear_cache

benchmark_throughput(
    predict_fn,
    clear_cache_fn,
    NODEL_NAME,
    get_sample_from_row_original,
    throughput_csvs,
    throughput_table,
    "fp32",
    False,
)

## fp32; + caching

predict_fn = lambda texts: gen_model.predict_fast(texts, batch_size=BATCH_SIZE)
clear_cache_fn = gen_model.clear_cache

benchmark_throughput(
    predict_fn,
    clear_cache_fn,
    NODEL_NAME,
    get_sample_from_row_original,
    throughput_csvs,
    throughput_table,
    "fp32",
    False,
)

## fp16; no caching
# TODO: reinit model?

## fp16; + caching
