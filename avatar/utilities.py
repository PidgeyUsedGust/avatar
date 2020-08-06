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


def to_mercs(df: pd.DataFrame) -> Tuple[np.ndarray,]:
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
    return new.values, nom
