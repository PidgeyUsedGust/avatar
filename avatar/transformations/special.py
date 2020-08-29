import pandas as pd
from typing import List, Tuple
from .base import Transformation


class Drop(Transformation):
    """Generic transformation."""

    allowed = []

    def __str__(self) -> str:
        return "{}()".format(self.__class__.__name__)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.DataFrame()

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        return [()]
