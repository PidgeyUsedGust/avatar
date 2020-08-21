"""Feature selection.

Two types of selection are performed.

 * Preselectors will use intrinsic properties of a single column
   in order to quickly remove columns for consideration.

"""
import pandas as pd
from pandas._typing import Label
from typing import Union, Optional


class StackedPreselector:
    """Combine differenct selectors."""

    def __init__(self, selectors):
        self._selectors = selectors

    def select(self, df: pd.DataFrame):
        for selector in self._selectors:
            df = selector.select(df)
        return df


class MissingPreselector:
    """Remove columns missing at least a percentage of values."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = 0.5

    def select(self, df: pd.DataFrame):
        return df.dropna(axis=1, thresh=(self._threshold * len(df.index)))


class ConstantPreselector:
    """Remove columns with constants."""

    def select(self, df: pd.DataFrame):
        return df.loc[:, (df != df.iloc[0]).any()]


class IdenticalPreselector:
    """Remove identical columns."""

    def select(self, df: pd.DataFrame):
        return df.drop(self.duplicates(df), axis=1)

    def duplicates(self, df: pd.DataFrame):
        duplicates = set()
        for i in range(df.shape[1]):
            col_one = df.iloc[:, i]
            for j in range(i + 1, df.shape[1]):
                col_two = df.iloc[:, j]
                if col_one.equals(col_two):
                    duplicates.add(df.columns[j])
                    break
        return duplicates


class UniquePreselector:
    """Remove columns containing only categorical, unique elements."""

    def select(self, df: pd.DataFrame):
        uniques = list()
        for column in df:
            if df[column].dtype.name in ["object", "category"]:
                if df[column].dropna().is_unique:
                    uniques.append(column)
        return df.drop(uniques, axis=1)
