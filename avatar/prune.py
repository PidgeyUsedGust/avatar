"""Prune columns.

Different pruners can be stacked together.

"""
import pandas as pd
from typing import List


class Pruner:
    """Generic column pruner."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class CADPruner(Pruner):
    """Categorical AllDifferent pruner.
    
    Removes columns that are categorical but consist of
    too many classes.

    Rather than only a true all different, the threshold
    can be changed to prune all columns that have more
    than `threshold * len(column)` different values (not
    counting NaN).

    """

    def __init__(self, threshold: float = 1.0):
        """

        Args:
            threshold: Percentage of values that need to be
                different for a categorical column to be
                pruned.

        """
        self._threshold = threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df:
            if df[column].dtype == "object":
                values = df[column].dropna()
                if values.nunique() >= len(values) * self._threshold:
                    df = df.drop(column, axis=1)
        return df


class NaNPruner(Pruner):
    """Prune columns with too many NaN values."""

    def __init__(self, threshold: float = 0.9):
        self._threshold = threshold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = len(df.index) * self._threshold
        for column in df:
            n_nan = df[column].isna().sum()
            if n_nan > threshold:
                df = df.drop(column, axis=1)
        return df


class DFPruner(Pruner):
    """Prune columns that are functionally dependent.
    
    If a categorical functional dependency is detected, keep only
    one of the columns.
    
    """

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class StackedPruner(Pruner):
    """Multiple pruners."""

    def __init__(self, pruners: List[Pruner]):
        self._pruners = pruners

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for pruner in self._pruners:
            df = pruner(df)
        return df
