"""Semantic transformations.

These might require external dependencies and are only
used if the dependencies are found.

"""
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
from .base import Transformation

from dateparser import parse
from word2number import w2n


def to_number(value: str) -> float:
    """Convert value to a number.

    Args:
        value: A string.

    Returns:
        Numeric value of string. If cannot convert, return `np.nan`.
    """
    try:
        number = w2n.word_to_num(value)
    except:
        number = np.nan
    return number


class WordToNumber(Transformation):
    """Convert words to numbers."""

    allowed = ["object", "category"]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.map(to_number).to_frame()

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        for value in column.drop_duplicates():
            v = to_number(value)
            if not np.isnan(v):
                return [()]
        return []


class NormaliseTimedelta(Transformation):
    """Normalise time deltas."""

    allowed = ["object", "category"]
    time_map = {"day": 1, "week": 7, "month": 30, "year": 365}

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        pass

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        values = np.unique(column)
        times = cls.time_map.keys()
        if any(any(t in value for t in times) for value in values):
            return [()]
        return []


class TimeFeatures(Transformation):
    """Date features."""

    allowed = ["object", "category"]
    directives = [
        "%A",
        "%w",
        "%-d",
        "%B",
        "%-m",
        "%-y",
        "%Y",
        "%-H",
        "%-I",
        "%p",
        "%-M",
        "%-S",
        "%f",
        "%z",
        "%Z",
        "%-j",
        "%U",
        "%W",
    ]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        datetimes = column.apply(parse)
        features = list()
        for directive in self.directives:
            new = datetimes.apply(lambda x: x.strftime(directive)).rename(directive[1:])
            if new.str.isnumeric().all():
                new = pd.to_numeric(new)
            features.append(new)
        return pd.concat(features, axis=1)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Any]]:
        head = column.head(100)
        if head.apply(parse).isna().sum() < (len(head) // 2):
            return [()]
        return []