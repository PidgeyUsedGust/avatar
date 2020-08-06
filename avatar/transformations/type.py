import pandas as pd
from typing import List, Tuple
from .base import Transformation


class Numerical(Transformation):
    """Make a column numerical."""

    allowed = ["object", "category"]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.to_numeric(column, errors="coerce").to_frame()

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        if column.str.isnumeric().all():
            return [()]
        return []


class Categorical(Transformation):
    """Make a column nominal (categorical)."""

    allowed = ["int64", "object"]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        # return pd.Categorical(column).to_frame()
        return column.astype("category").to_frame()

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        if column.duplicated().any():
            return [()]
        return []
