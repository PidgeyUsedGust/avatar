import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from .base import Transformation


class OneHot(Transformation):
    """One hot encode a column."""

    allowed = ["object", "int64", "category"]

    threshold: Union[float, int] = 20
    """Maximal number of categories."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(column)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        """Require at least one duplicate value."""
        threshold = len(column) * cls.threshold if cls.threshold < 1 else cls.threshold
        if column.nunique() < threshold:
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
        self._value = value

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.replace(self._value, np.nan).to_frame()

    def __str__(self) -> str:
        return "NaN({})".format(self._value)

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
