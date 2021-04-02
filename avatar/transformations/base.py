"""Base transformation."""
import pandas as pd
from typing import List, Tuple, Any


class Transformation:
    """Generic transformation."""

    allowed = []

    def __str__(self) -> str:
        return "{}()".format(self.__class__.__name__)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(column)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Any]]:
        return ()

    # @property
    # def name(self) -> str:
    #     return self.__class__.__name__