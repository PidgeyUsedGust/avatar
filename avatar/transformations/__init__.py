from typing import Tuple, Any
import pandas as pd


class Transformation:
    """Generic transformation."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(column)

    @classmethod
    def arguments(self, column: pd.Series) -> Tuple[Any]:
        return ()