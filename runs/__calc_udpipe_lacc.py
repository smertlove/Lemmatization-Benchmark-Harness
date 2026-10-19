
"""

UDPipe не работает в нашей постановке, поэтому замер производим вот такой кривенький :(
"""

MODEL_PATH = "/mnt/data_storage/models/ud-models/russian-syntagrus-ud-2.5-191206.udpipe"


from pathlib import Path
from typing import TypedDict, Iterator
from conllu import parse_incr


def _iter_dataset_dir(
    datadir: Path | str,
    ignoredir_pref: str = "__",
    valid_subsets: set[str] = {"train", "test", "dev"},
) -> Iterator[tuple[str, str, Path]]:

    if isinstance(datadir, str):
        datadir = Path(datadir)
    assert datadir.exists()

    for dir_ in sorted(datadir.iterdir(), key=lambda dir_: dir_.name):

        # Ignore root files and dirs starting with ignoredir_pref
        if not dir_.is_dir() or dir_.name.startswith(ignoredir_pref):
            continue

        # Delete all splits with weird naming
        children = [
            subdir_
            for subdir_ in dir_.iterdir()
            if subdir_.name in valid_subsets and subdir_.is_dir()
        ]

        for child in children:
            yield dir_.name, child.name, child


def get_train_dev_test_paths(
    datadir: Path | str,
):
    it = _iter_dataset_dir(datadir)

    splits = {
        "train": [],
        "test": [],
        "dev": [],
    }

    for subset, split, dir_ in it:
        for file in dir_.iterdir():
            if file.suffix != ".conllu":
                raise ValueError(f"{file} extension '{file.suffix}' != '.conllu'")
            splits[split].append(file)

    return splits


DATA_PATH = "/mnt/data_storage/datasets/conllu/rubic_data-master"

splits = get_train_dev_test_paths(DATA_PATH)


def _parse_sentences(paths):
    for file_path in paths:

        with open(file_path, "r", encoding="utf-8") as f:
            for sentence in parse_incr(f):
                yield sentence


it = list(_parse_sentences(splits["test"]))


import spacy_udpipe
from spacy.tokens import DocBin

nlp = spacy_udpipe.load_from_path(
    lang="ru",
    path=MODEL_PATH,
)

from tqdm import tqdm

actual = []
predicted = []
misaligned = 0
good = 0
for sent in tqdm(it):
    text = sent.metadata["text"]
    doc = nlp(text)

    doc_forms = [token.text for token in doc]
    real_forms = [token["form"] for token in sent]

    if doc_forms != real_forms:
        misaligned += 1
        continue

    good += 1
    predicted.extend([token.lemma_ for token in doc])

    actual.extend([token["lemma"] for token in sent])

import pandas as pd

df = pd.DataFrame()
df["actual"] = actual
df["predicted"] = predicted

print(df.shape, misaligned, good)

from src.metrics import Calculator

calculator = Calculator()

print(
    f"lAcc {calculator.lAcc(df["predicted"], df["actual"]):.4f}",  
    f"lAcc (norm) {calculator.lAcc(df["predicted"], df["actual"], normalize=True):.4f}", 
)


