"""Utility functions."""
from typing import Sequence, List, Tuple
import pandas as pd
import numpy as np


def get_substrings(string: Sequence) -> List[Sequence]:
    """Get substrings of a sequence."""
    substrings = list()
    for i in range(1, len(string) + 1):
        for j in range(0, len(string) - i + 1):
            substring = string[j : j + i]
            if substring not in substrings:
                substrings.append(substring)
    return substrings


def count_unique(df: pd.DataFrame) -> int:
    """Count number of unique values in a dataframe.
    
    TODO — Perform analysis of faster alternatives.

    """
    return pd.unique(df.values.ravel()).size


def normalize(s: pd.Series) -> pd.Series:
    """Min-max scale."""
    mi = s.min()
    ma = s.max()
    if mi == ma:
        if ma == 0:
            return s
        return s / ma
    return (s - s.min()) / (s.max() - s.min())


def xor(a: bool, b: bool) -> bool:
    return (a and b) or (not a and not b)