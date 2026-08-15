from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from cachetools import LRUCache

from tqdm import tqdm
import math
import torch

from typing import Callable

Heuristic = Callable[[str], str | None]


class GenerativeModel:

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device="cuda",
    ):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def model_preproc(self, texts: list[str]):
        """
        Предобрабатывает входные тексты для модели
        """
        return texts

    def model_postproc(self, texts: list[str]):
        """
        Постобрабатывает гипотезы модели
        """
        return texts

    def predict(
        self,
        texts: list[str],
        max_length=32,
        batch_size=32,
        verbose=True,
    ) -> list[str]:
        """
        Предобрабатывает тексты, генерирует леммы и постобрабатывает их.
        """

        texts = self.model_preproc(texts)

        preds = self._predict(
            texts, max_length=max_length, batch_size=batch_size, verbose=verbose
        )

        preds = self.model_postproc(preds)

        return preds

    def _predict(
        self,
        texts: list[str],
        max_length: int = 32,
        batch_size: int = 32,
        verbose=True,
    ) -> list[str]:

        preds = []

        if verbose:
            pbar: range | tqdm[int] = tqdm(
                range(0, len(texts), batch_size),
                total=math.ceil(len(texts) / batch_size),
            )
        else:
            pbar = range(0, len(texts), batch_size)

        for i in pbar:
            batch = texts[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=70,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=1,
                    temperature=1,
                    early_stopping=True,
                )

            cur_preds = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            preds.extend(cur_preds)

        return preds


class GenerativeModelWithCachingAndHeuristics(GenerativeModel):

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device="cuda",
        cache_size=100000,
        heuristics_stack: list[Heuristic] | None = None,
    ):
        super().__init__(model, tokenizer, device)

        self._cache: LRUCache = LRUCache(maxsize=cache_size)
        self._heuristics_stack = heuristics_stack if heuristics_stack is not None else list()

    def _add_to_global_cache(self, key: str, val: str):
        self._cache[key] = val

    def _get_from_global_cache(self, key: str):
        return self._cache.get(key)

    def clear_cache(self):
        self._cache.clear()

    def _apply_heuristics(self, inpt: str) -> str | None:
        """
        Применяет эвристики. Кэширует в случае если какая-то эвристика успешно применяется.
        """
        for heuristic in self._heuristics_stack:
            heuristic_pred = heuristic(inpt)
            if heuristic_pred is not None:
                self._add_to_global_cache(inpt, heuristic_pred)
                return heuristic_pred
        return None

    def _fast_get_or_None(self, inpt: str) -> str | None:
        """
        Берет из кеша, либо применяет эвристики
        """
        for step in (
            self._get_from_global_cache,
            self._apply_heuristics,
        ):
            result = step(inpt)
            if result is not None:
                return result
        return None

    def _exec_model_predict(
        self,
        batch: list[str],
        max_length=32,
        batch_size=32,
    ):
        """
        Генерирует леммы при помощи модели и заполняет кеш
        """
        uniq_inpts = list(set(batch))
        preds = self.predict(
            uniq_inpts, max_length=max_length, batch_size=batch_size, verbose=False
        )

        # Локальный кеш
        loc_cache = {inpt: pred for inpt, pred in zip(uniq_inpts, preds)}

        result = []
        for inpt in batch:
            pred = loc_cache[inpt]
            result.append(pred)

            # Глобальный кэш может вести себя по-разному в зависимости от имплементации,
            # поэтому мы заполняем его отдельным методом
            self._add_to_global_cache(inpt, pred)

        return result

    def _predict_batch(
        self,
        batch: list[str],
        max_length=32,
        batch_size=32,
    ):

        result = [self._fast_get_or_None(inpt) for inpt in batch]
        pruned_batch = [inpt for inpt, res in zip(batch, result) if res is None]

        if pruned_batch:
            preds = self._exec_model_predict(
                pruned_batch,
                max_length=max_length,
                batch_size=batch_size,
            )

            preds_ptr = 0
            for i in range(len(result)):
                if result[i] is None:
                    result[i] = preds[preds_ptr]
                    preds_ptr += 1

        return result

    def predict_fast(
        self,
        inpts: list[str],
        max_length=32,
        batch_size=32,
        verbose=True,
    ) -> list[str]:
        """
        Быстрый `.predict` с привлечением кеша и эвристик
        """
        result = []

        ln = len(inpts)
        bs = (
            batch_size * 3
        )  # Нужно чтобы на модель отправлялся примерно нужный батчсайз после кеша

        if verbose:
            tot = math.ceil(ln / bs)
            pbar: tqdm[int] | range = tqdm(range(0, ln, bs), total=tot)
        else:
            pbar = range(0, ln, bs)

        for i in pbar:
            batch = inpts[i : i + bs]
            preds = self._predict_batch(
                batch, max_length=max_length, batch_size=batch_size
            )
            result.extend(preds)

        return result

__all__ = (
    "GenerativeModel",
    "GenerativeModelWithCachingAndHeuristics",
)
