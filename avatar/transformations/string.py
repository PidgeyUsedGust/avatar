"""String transformations."""
import string
import itertools
from typing import Tuple, Any, List
import pandas as pd

from . import Transformation
from ..utilities import get_substrings


class Split(Transformation):
    """Split column by delimiter."""

    def __init__(self, delimiter: str):
        self._delimiter = delimiter

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.split(pat=self._delimiter, expand=True)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Get possible delimiters.
        
        Get all substrings of non-alphanumeric characters.

        """
        # figure out all non alphanumeric strings and deduplicate them.
        split = column.str.split(pat=r"[a-zA-Z0-9]+")
        split = split[split.astype(str).drop_duplicates().index]
        # generate delimiters by taking consecutive
        arguments = set()
        for delimiters in split:
            delimiters = [d for d in delimiters if d]
            for delimiter in delimiters:
                for substring in get_substrings(delimiter):
                    arguments.add((substring,))
        return arguments


class SplitAlign(Transformation):
    """Split by delimiter and align by column.
    
    For example, a column 

        A,B
        B,
        A,C
    
    gets split into

        A B
          B
        A   C
    
    which allows every token to be considered as a feature.

    """

    def __init__(self, delimiter: str):
        self._delimiter = delimiter

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame()
        for i, values in column.str.split(pat=self._delimiter).iteritems():
            for value in values:
                df.loc[i, value] = value
        return df.fillna("")

    # def __call__(self, column: pd.Series) -> pd.DataFrame:
    #     return column.str.get_dummies(self._delimiter)

    @classmethod
    def arguments(cls, column: pd.Series) -> Tuple[Any]:
        """Same arguments as regular split."""
        Split.arguments(column)


class Extract(Transformation):
    """Extract a regex pattern from a string."""

    def __init__(self, pattern: str):
        self._pattern = pattern

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.extract(pat=self._pattern, expand=True)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        pass