import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from .base import Transformation


class Dummies(Transformation):
    """One hot encode a column."""

    allowed = ["string", "int"]
    max_categories: Union[float, int] = 20
    """Maximal number of categories."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(column)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        """Require at least three and at most most `threshold` categories"""
        unique = column.nunique()
        if unique <= cls.max_categories and unique > 2:
            return [()]
        return []


class NaN(Transformation):
    """Encode a new value as NaN."""

    allowed = ["object", "category"]
    trigger = [
        "unknown",
        "unknwon",
        "some",
        "not",
        "fake",
        "any",
        "?",
        "0000",
        "1111",
        "9999",
    ]
    """List of DMV triggers.

    We use a list of trigger values to detect possible
    disguised missing values. These are obtained from
    experience and polls on Reddit an Kaggle.

    """

    def __init__(self, value: str):
        self.value = value

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.replace(self.value, np.nan).to_frame()

    def __str__(self) -> str:
        return "NaN({})".format(self.value)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Heuristic."""
        arguments = list()
        for value in column.drop_duplicates().dropna():
            if cls.dmv(value):
                arguments.append((value,))
        return arguments

    @classmethod
    def dmv(cls, value: str) -> bool:
        """Check if value is a disguised missing value."""
        return any(trigger in value.lower() for trigger in cls.trigger)
