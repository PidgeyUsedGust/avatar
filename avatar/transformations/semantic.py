"""Semantic transformations.

These might require external dependencies and are only
used if the dependencies are found.

"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from .base import Transformation


# try:
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


# except:
#     pass
