"""Imputation transformations."""
import pandas as pd
import numpy as np
from typing import List, Tuple
from pandas.api.types import is_numeric_dtype
from .base import Transformation


class MeanImputation(Transformation):

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.fillna(column.mean()).to_frame()
    
    @classmethod
    def arguments(self, column: pd.Series) -> List[Tuple[()]]:
        if is_numeric_dtype(column) and column.isna().any():
            return [()]
        return []


class MedianImputation(Transformation):

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.fillna(column.median()).to_frame()
    
    @classmethod
    def arguments(self, column: pd.Series) -> List[Tuple[()]]:
        if is_numeric_dtype(column) and column.isna().any():
            return [()]
        return []


class ModeImputation(Transformation):

    allowed = ["object", "category"]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.fillna(column.mode()[0]).to_frame()
    
    @classmethod
    def arguments(self, column: pd.Series) -> List[Tuple[()]]:
        if column.isna().any():
            return [()]
        return []