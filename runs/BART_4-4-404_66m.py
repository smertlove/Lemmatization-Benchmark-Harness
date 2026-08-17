import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from src import (
    GenerativeModelWithCachingAndHeuristics,
    benchmark_lemmatization_quality,
    benchmark_throughput,
    get_sample_from_row_original,
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

MODEL_NAME = "BART_4-4-404_66m"
MODEL_ID = models_dir / MODEL_NAME
BATCH_SIZE = 512
CACHE_SIZE = 5_000_000


class CharLevelGenerativeModel(GenerativeModelWithCachingAndHeuristics):

    def __init__(self, model, tokenizer, device="cuda", cache_size=100000, heuristics_stack = None):
        super().__init__(model, tokenizer, device, cache_size, heuristics_stack)
        self.bos = self.tokenizer.bos_token or ""
        self.eos = self.tokenizer.eos_token or ""

    def model_preproc(self, texts):
        return [self.bos + text + self.eos for text in texts]

    def model_postproc(self, texts):
        return ["".join(text.split()) for text in texts]


def load_gen_model(dtype: str) -> CharLevelGenerativeModel:
    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    model = BartForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return CharLevelGenerativeModel(
        model, tokenizer, cache_size=CACHE_SIZE
    )


def release_gen_model(gen_model: CharLevelGenerativeModel) -> None:
    del gen_model.model, gen_model.tokenizer, gen_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_throughput_benchmark(
    gen_model: CharLevelGenerativeModel,
    dtype: str,
    caching: bool,
) -> None:
    if caching:
        predict_fn = lambda texts: gen_model.predict_fast(texts, batch_size=BATCH_SIZE)
    else:
        predict_fn = lambda texts: gen_model.predict(texts, batch_size=BATCH_SIZE)

    benchmark_throughput(
        predict_fn,
        gen_model.clear_cache,
        MODEL_NAME,
        get_sample_from_row_original,
        throughput_csvs,
        throughput_table,
        dtype,
        caching,
    )


def run_quality_benchmark(gen_model: CharLevelGenerativeModel) -> None:
    predict_fn = lambda texts: gen_model.predict_fast(texts, batch_size=BATCH_SIZE)

    benchmark_lemmatization_quality(
        predict_fn,
        MODEL_NAME,
        get_sample_from_row_original,
        quality_csvs,
        quality_table,
    )


if __name__ == "__main__":

    ## ==== fp32 ==== ##

    gen_model = load_gen_model("fp32")
    sanity_check(gen_model)

    run_throughput_benchmark(gen_model, dtype="fp32", caching=False)
    run_throughput_benchmark(gen_model, dtype="fp32", caching=True)

    release_gen_model(gen_model)

    ## ==== fp16 ==== ##

    gen_model = load_gen_model("fp16")

    run_throughput_benchmark(gen_model, dtype="fp16", caching=False)
    run_throughput_benchmark(gen_model, dtype="fp16", caching=True)

    run_quality_benchmark(gen_model)

    release_gen_model(gen_model)
