"""Simple transformation language."""
import pandas as pd
from pandas._typing import Label
from typing import List, Tuple, Optional
from .transformations import *


default_transformations = [
    # string
    Split,
    SplitAlign,
    ExtractNumber,
    ExtractWord,
    Lowercase,
    # type
    Numerical,
    # encoding
    OneHot,
    NaN,
    # semantic
    WordToNumber,
    # imputation
    MeanImputation,
    ModeImputation,
    MedianImputation,
]
"""List of all supported transformations."""


class WranglingTransformation:
    """Generic wrangling transformation."""

    def __init__(self, column, transformation: Transformation, replace=False):
        """
        
        Args:
            replace: If set to True, will remove transformed column.

        """
        self._column = column
        self._transformation = transformation
        self._replace = replace

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to dataframe.
        
        If replacing is True, remove the original column.

        """
        new_columns = self._transformation(df[self._column])
        new_columns.rename(
            columns=lambda x: "{}({})_{}".format(self._transformation, self._column, x),
            inplace=True,
        )
        # new_columns.columns = [str(self._transformation) + ]
        new_df = pd.concat((df, new_columns), axis=1)
        if self._replace:
            new_df = new_df.drop(self._column)
        return new_df

    def __str__(self):
        return "{}({})".format(self._transformation, self._column)


class WranglingLanguage:
    """Wrangling language."""

    def __init__(self, transformations: Optional[List[Transformation]] = None):
        """

        Arguments:
            transformations: A list of transformations supported by
                the current language.

        """
        if transformations is None:
            transformations = default_transformations
        self._transformations = transformations

    def transformations(
        self,
        df: pd.DataFrame,
        exclude: Optional[List[Label]] = None,
        target: Label = None,
    ) -> List[Tuple[str, Transformation]]:
        """Get allowed arguments for wrangling transformation.
        
        Args:
            exclude: Subset of columns to exclude.
    
        """
        if exclude is None:
            exclude = set()
        if target is not None:
            exclude.add(target)
        transformations = list()
        for transformation in self._transformations:
            for i, column in df.iteritems():
                if i not in exclude and column.dtype.name in transformation.allowed:
                    arguments = transformation.arguments(column)
                    for argument in arguments:
                        transformations.append(
                            WranglingTransformation(i, transformation(*argument))
                        )
        return transformations

    def expand(
        self,
        df: pd.DataFrame,
        exclude: Optional[List[Label]] = None,
        target: Label = None,
    ) -> pd.DataFrame:
        """Expand dataframe with all transformations.
        
        Args:
            exclude: Subset of columns to exclude.
            target: If provided, will skip this column.

        """
        transformations = self.transformations(df, exclude=exclude, target=target)
        for transformation in transformations:
            df = transformation(df)
        return df


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
