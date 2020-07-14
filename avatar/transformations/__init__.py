from typing import Tuple, List, Any
import pandas as pd


class Transformation:
    """Generic transformation."""

    def __str__(self) -> str:
        return self.__class__.__name__

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(column)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Any]]:
        return ()


class Filter:
    """Generic filter."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def keep(self, value):
        return True
    
    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Any]]:
        return []