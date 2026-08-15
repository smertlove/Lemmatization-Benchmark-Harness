# Lemmatization-Benchmark-Harness
Обвязка для замеров качества/быстродействия генеративных моделей для задач лемматизации 

```bash
uv run python -m run_bench \
  --quality-csvs /mnt/data_storage/datasets/generative_lemmatization_datasets/csvs/bench/* \
  --speed-csvs /mnt/data_storage/datasets/generative_lemmatization_datasets/csvs/bench/test.csv

```

