import pandas as pd
import numpy as np
from typing import List, Tuple
from .base import Transformation


class OneHot(Transformation):
    """One hot encode a column."""

    allowed = ["object", "int64", "category"]

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(column)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        """Require at least one duplicate value."""
        if column.duplicated().any():
            return [()]
        return []


class NaN(Transformation):
    """Encode a new value as NA."""

    allowed = ["object", "category"]

    def __init__(self, value: str):
        self._value = value

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.replace(self._value, np.nan).to_frame()

    def __str__(self) -> str:
        return "NaN({})".format(self._value)

    @classmethod
    def arguments(self, column: pd.Series) -> List[Tuple[str]]:
        arguments = list()
        for value, count in column.value_counts().iteritems():
            if count > 1:
                arguments.append((value,))
            else:
                break
        return arguments
        # return [(v,) for v in column.drop_duplicates().to_list()]
