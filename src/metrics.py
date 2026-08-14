import pymorphy3 as pm
from functools import lru_cache
from jiwer import cer
import numpy as np


class _Lemmatizer:

    def __init__(self):
        self.morph = pm.MorphAnalyzer()

    @lru_cache(123123123)
    def lemmatize(self, word: str):
        return self.morph.parse(word)[0].normal_form


class Calculator:

    def __init__(self):
        self.lemmatizer = _Lemmatizer()

    def _preproc(self, words: list[str]):
        return [
            word.lower().replace("ё", "е")
            for word in words
        ]

    def _normalize(self, words: list[str]):
        return [
            self.lemmatizer.lemmatize(word)
            for word in words
        ]

    def _conditional_conversion(self, y: str|list[str], y_bar:str|list[str], normalize=False):
        if isinstance(y, str):
            y = [y]

        if isinstance(y_bar, str):
            y_bar = [y_bar]

        assert len(y) == len(y_bar)

        y = self._preproc(y)
        y_bar = self._preproc(y_bar)

        if normalize:
            y = self._normalize(y)
            y_bar = self._normalize(y_bar) 

        return y, y_bar

    def _lAcc(self, y: list[str], y_bar:list[str]):

        hits = 0
        for target, pred in zip(y, y_bar):
            if target == pred:
                hits += 1

        return hits / len(y)
        
    def lAcc(self, y: str|list[str], y_bar:str|list[str], normalize=False):
        """
            Lemmatization accuracy for Russian
            disregarding е/ё choice and upper/lower case
        """
        y, y_bar = self._conditional_conversion(y, y_bar)
        return self._lAcc(y, y_bar)

    def _CER(self, y: list[str], y_bar:list[str]):
        return cer(y, y_bar)

    def CER(self, y: str|list[str], y_bar:str|list[str], normalize=False):
        """
            Lemmatization CER for Russian
            disregarding е/ё choice and upper/lower case
        """
        y, y_bar = self._conditional_conversion(y, y_bar)
        return self._CER(y, y_bar)
