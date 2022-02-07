"""Simple transformation language."""
import pandas as pd
from typing import List, Optional, Type
from .transformations import *
from .utilities import encode_name


default_transformations = [
    Split,
    SplitDummies,
    ExtractNumberPattern,
    ExtractInteger,
    ExtractBoolean,
    WordToNumber,
    Dummies,
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
        self.transformation = transformation
        self._replace = replace

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        new_columns = self.transformation(df[self._column])
        new_columns.rename(
            columns=lambda x: encode_name("{}_{}".format(self, x)),
            inplace=True,
        )
        return new_columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to dataframe.

        If replacing is True, remove the original column.

        """
        new_columns = self.execute(df)
        new_df = pd.concat((df, new_columns), axis=1)
        if self._replace and (self._column in new_df.columns):
            new_df = new_df.drop(self._column, axis=1)
        return new_df

    def __eq__(self, other: "WranglingTransformation"):
        return (self._column == other._column) and (
            self.transformation == other.transformation
        )

    def __str__(self):
        return "{}({})".format(self.transformation, self._column)

    def __repr__(self):
        return "WranglingTransformation({}, {})".format(
            self._column, self.transformation
        )


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

    def __repr__(self) -> str:
        return str(self)

    def grow(self, transformation: WranglingTransformation):
        self._transformations.append(transformation)

    @property
    def transformations(self):
        return self._transformations


class WranglingLanguage:
    """Wrangling language."""

    def __init__(self, transformations: Optional[List[Type[Transformation]]] = None):
        """

        Arguments:
            transformations: A list of transformations supported by
                the current language.

        """
        if transformations is None:
            transformations = default_transformations
        self.transformations = transformations

    def get_transformations(self, column: pd.Series) -> List[WranglingTransformation]:
        """Get transformations that can be applied to column.

        Returns:
            A list of transformations.

        """
        transformations = list()
        for transformation in self.transformations:
            if column.dtype.name in transformation.allowed:
                arguments = transformation.arguments(column)
                for argument in arguments:
                    transformations.append(
                        WranglingTransformation(column.name, transformation(*argument))
                    )
        return transformations

    def reset(self) -> None:
        """Reset all transformation caches."""
        for transformation in self.transformations:
            if hasattr(transformation, "_cache"):
                transformation._cache = dict()

    def __str__(self) -> str:
        return ", ".join(map(lambda t: str(t.__name__), self.transformations))

    # def transformations(
    #     self,
    #     df: pd.DataFrame,
    #     target: Hashable = None,
    #     exclude: Optional[List[Hashable]] = None,
    # ) -> List[Tuple[str, Transformation]]:
    #     """Get allowed arguments for wrangling transformation.

    #     Args:
    #         exclude: Subset of columns to exclude.

    #     """
    #     exclude = exclude or set()
    #     exclude.add(target)
    #     pbar = tqdm.tqdm(
    #         total=len(self._transformations) * len(df.columns),
    #         disable=not verbose,
    #         desc="Finding  transformations",
    #     )
    #     # find transformations
    #     transformations = list()
    #     for transformation in self._transformations:
    #         for i, column in df.iteritems():
    #             if i not in exclude and column.dtype.name in transformation.allowed:
    #                 arguments = transformation.arguments(column)
    #                 for argument in arguments:
    #                     transformations.append(
    #                         WranglingTransformation(i, transformation(*argument))
    #                     )
    #             pbar.update()
    #     # store transformations
    #     for transformation in transformations:
    #         self._transformation_map[str(transformation)] = transformation
    #     return transformations

    # def expand(
    #     self,
    #     df: pd.DataFrame,
    #     exclude: Optional[List[Hashable]] = None,
    #     target: Hashable = None,
    # ) -> pd.DataFrame:
    #     """Expand dataframe with all transformations.

    #     Args:
    #         exclude: Subset of columns to exclude.
    #         target: If provided, will skip this column.

    #     """
    #     transformations = self.transformations(df, exclude=exclude, target=target)
    #     dataframes = [df]
    #     columns = set(df.columns)
    #     pbar = tqdm.tqdm(
    #         total=len(transformations),
    #         disable=not verbose,
    #         desc="Applying transformations",
    #     )
    #     for transformation in transformations:
    #         new_dataframe = transformation.execute(df)
    #         new_columns = set(new_dataframe.columns)
    #         if not (columns & new_columns):
    #             dataframes.append(new_dataframe)
    #             columns.update(new_columns)
    #         pbar.update()
    #     return pd.concat(dataframes, axis=1).replace("", np.nan)

    # def make_program(
    #     self, df: pd.DataFrame, features: List[Hashable]
    # ) -> "WranglingProgram":
    #     """Build program from selected features.

    #     Args:
    #         df: Original dataframe.

    #     """
    #     # keep a queue of features to find and their depth
    #     queue = [(feature,) for feature in features]
    #     chains = list()
    #     # find transformations
    #     while len(queue) > 0:
    #         chain = queue.pop()
    #         # get name of column
    #         feature = chain[-1]
    #         if isinstance(feature, str):
    #             chain = ()
    #         else:
    #             feature = feature._column
    #         # find transformation that made feature
    #         found = None
    #         for name, transformation in self._transformation_map.items():
    #             if feature.startswith(name):
    #                 found = transformation
    #         # add next to queue
    #         if found is not None:
    #             queue.append((*chain, found))
    #         else:
    #             chains.append(chain)
    #     # build into program
    #     program = WranglingProgram()
    #     for chain in chains:
    #         chain = chain[::-1]
    #         for transformation in chain:
    #             if transformation not in program:
    #                 program.grow(transformation)
    #     for column in program(df).columns:
    #         if column not in features:
    #             program.grow(WranglingTransformation(column, Drop(), replace=True))
    #     return program
