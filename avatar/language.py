"""Simple transformation language."""
import pandas as pd
from transformations import Transformation


class WranglingTransformation:
    """Generic wrangling transformation."""

    def __init__(self, column: int, transformation: Transformation):
        self._column = column
        self._transformation = transformation

    @classmethod
    def arguments(self, df: pd.DataFrame) -> List[Tuple[int, Transformation]]:
        """Get allowed arguments for wrangling transformation."""
        for i, column in df.iteritems():
            print(i, column)


class WranglingProgram:
    """Wrangling program."""

    def __init__(self):
        self._transformations = list()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run transformation on a dataframe."""
        for transformation in self._transformations:
            df = transformation(df)
        return df

    def grow(self, transformation: WranglingTransformation):
        self._transformations.append(transformation)
