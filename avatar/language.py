"""Simple transformation language."""
import pandas as pd
import tqdm
from pandas._typing import Label
from typing import List, Tuple, Optional
from .transformations import *
from .settings import verbose


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
            columns=lambda x: "{}_{}".format(self, x),
            inplace=True,
        )
        new_df = pd.concat((df, new_columns), axis=1)
        if self._replace and (self._column in new_df.columns):
            new_df = new_df.drop(self._column, axis=1)
        return new_df

    def __eq__(self, other: "WranglingTransformation"):
        return (self._column == other._column) and (
            self._transformation == other._transformation
        )

    def __str__(self):
        return "{}({})".format(self._transformation, self._column)


class WranglingProgram:
    """Wrangling program."""

    def __init__(self):
        self._transformations = list()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for transformation in self._transformations:
            df = transformation(df)
        return df

    def __contains__(self, transformation: "WranglingTransformation"):
        return transformation in self._transformations

    def __str__(self) -> str:
        return "\n".join(map(str, self._transformations))

    def grow(self, transformation: WranglingTransformation):
        self._transformations.append(transformation)


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
        self._transformation_map = dict()

    def transformations(
        self,
        df: pd.DataFrame,
        target: Label = None,
        exclude: Optional[List[Label]] = None,
    ) -> List[Tuple[str, Transformation]]:
        """Get allowed arguments for wrangling transformation.

        Args:
            exclude: Subset of columns to exclude.

        """
        exclude = exclude or set()
        if target is not None:
            exclude.add(target)
        pbar = tqdm.tqdm(
            total=len(self._transformations) * len(df.columns),
            disable=not verbose,
            desc="Finding  transformations",
        )
        # find transformations
        transformations = list()
        for transformation in self._transformations:
            for i, column in df.iteritems():
                if i not in exclude and column.dtype.name in transformation.allowed:
                    arguments = transformation.arguments(column)
                    for argument in arguments:
                        transformations.append(
                            WranglingTransformation(i, transformation(*argument))
                        )
                pbar.update()
        # store transformations
        for transformation in transformations:
            self._transformation_map[str(transformation)] = transformation
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
        pb = tqdm.tqdm(
            total=len(transformations),
            disable=not verbose,
            desc="Applying transformations",
        )
        for transformation in transformations:
            df = transformation(df)
            pb.update()
        return df

    def make_program(
        self, df: pd.DataFrame, features: List[Label]
    ) -> "WranglingProgram":
        """Build program from selected features.

        Args:
            df: Original dataframe.

        """
        # keep a queue of features to find and their depth
        queue = [(feature,) for feature in features]
        chains = list()
        # find transformations
        while len(queue) > 0:
            chain = queue.pop()
            # get name of column
            feature = chain[-1]
            if isinstance(feature, str):
                chain = ()
            else:
                feature = feature._column
            # find transformation that made feature
            found = None
            for name, transformation in self._transformation_map.items():
                if feature.startswith(name):
                    found = transformation
            # add next to queue
            if found is not None:
                queue.append((*chain, found))
            else:
                chains.append(chain)
        # build into program
        program = WranglingProgram()
        for chain in chains:
            chain = chain[::-1]
            for transformation in chain:
                if transformation not in program:
                    program.grow(transformation)
        for column in program(df).columns:
            if column not in features:
                program.grow(WranglingTransformation(column, Drop(), replace=True))
        return program