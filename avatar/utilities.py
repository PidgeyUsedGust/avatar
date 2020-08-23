"""Utility functions."""
from typing import Sequence, List, Tuple, Set
from pandas._typing import Label
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


def to_mercs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[Label]]:
    """Encode dataframe for MERCS.
    
    We assume numerical values will have a numerical dtype and convert
    everything else to nominal integers.

    """
    new = pd.DataFrame()
    nom = set()
    for i, column in enumerate(df):
        if df[column].dtype.name in ["category", "object"]:
            new[column] = df[column].astype("category").cat.codes.replace(-1, np.nan)
            nom.add(i)
        else:
            new[column] = df[column]
    return new, nom


def to_m_codes(columns: pd.Index, target: Label):
    """Generate m_codes for a target."""
    if target is None:
        return None
    m_codes = np.zeros((1, len(columns)))
    m_codes[0, columns.get_loc(target)] = 1
    return m_codes
