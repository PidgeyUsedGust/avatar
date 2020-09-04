import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import abstractmethod, ABC
from pandas._typing import Label
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from mercs.core import Mercs
from .utilities import to_mercs
from .settings import verbose


class Filter(ABC):
    """Generic filter."""

    @abstractmethod
    def select(self, df: pd.DataFrame, target=None) -> pd.DataFrame:
        pass


class StackedFilter:
    """Combine differenct filters."""

    def __init__(self, selectors):
        self._selectors = selectors

    def select(self, df: pd.DataFrame, target=None):
        for selector in self._selectors:
            df = selector.select(df, target=target)
        return df


class MissingFilter:
    """Remove columns missing at least a percentage of values."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = 0.5

    def select(self, df: pd.DataFrame, target=None):
        return df.dropna(axis=1, thresh=(self._threshold * len(df.index)))


class GreedyMissingFilter:
    """Greedily remove column with most missing values."""

    def __init__(self, min_full: int = 10):
        self._min_full = min_full

    def select(self, df: pd.DataFrame, target: Label = None):
        m_count = df.isna().sum(axis=0)
        missing = m_count.index[(m_count.values).argsort()].tolist()
        while df.notna().all(axis=1).sum() < self._min_full:
            remove = missing.pop()
            df = df.drop(remove, axis=1)
        return df


class ConstantFilter:
    """Remove columns with constants."""

    def select(self, df: pd.DataFrame, target=None):
        return df.loc[:, (df != df.iloc[0]).any()]


# class IdenticalFilter:
#     """Remove identical columns.

#     Remove numerical columns that are identical.

#     """

#     def __init__(self, threshold: float = 0.98):
#         self._threshold = threshold

#     def select(self, df: pd.DataFrame, target=None):
#         return df.drop(self.duplicates(df), axis=1)

#     def duplicates(self, df: pd.DataFrame):
#         duplicates = set()
#         for i in range(df.shape[1]):
#             if i > len(df.columns):
#                 break
#             col_one = df.iloc[:, i]
#             for j in range(i + 1, df.shape[1]):
#                 col_two = df.iloc[:, j]
#                 hamming = (col_one == col_two).sum() / len(df.index)
#                 if hamming > self._threshold:
#                     duplicates.add(df.columns[j])
#         return duplicates


class IdenticalFilter:
    """Remove identical columns.

    Remove numerical columns that are identical.

    """

    def __init__(self, threshold: float = 0.98):
        self._threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        # first, remove exact duplicates
        df = df.loc[:, ~df.T.duplicated(keep="first")]
        # then, almost 
        pbar = tqdm(total=len(df.columns), disable=not verbose)
        for i in range(df.shape[1]):
            if i >= len(df.columns):
                break
            col_one = df.iloc[:, i]
            compare = df.iloc[:, i + 1 :].eq(col_one, axis=0)
            hamming = compare.sum() / len(df.index)
            df = df.drop(hamming[hamming > self._threshold].index, axis=1)
            pbar.update()
            pbar.set_description("ID " + str(len(df.columns)))
        return df


# class BijectiveFilter:
#     """Remove categorical columns that are (almost) bijective."""

#     def __init__(self, threshold: float = 0.95):
#         self._threshold = threshold

#     def select(self, df: pd.DataFrame, target=None):
#         return df.drop(self.duplicates(df), axis=1)

#     def duplicates(self, df: pd.DataFrame):
#         # select only object
#         df = df.select_dtypes(include="object")
#         duplicates = set()
#         for i in range(df.shape[1]):
#             col_one = df.iloc[:, i]
#             for j in range(i + 1, df.shape[1]):
#                 col_two = df.iloc[:, j]
#                 hamming = (
#                     col_one.factorize()[0] == col_two.factorize()[0]
#                 ).sum() / len(df.index)
#                 if hamming > self._threshold:
#                     duplicates.add(df.columns[j])
#         return duplicates


class BijectiveFilter:
    """Remove categorical columns that are (almost) bijective."""

    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        # get all objects
        dfo = df.select_dtypes(include="object")
        dfo_columns = dfo.columns
        # factorize whole object dataframe and remove exacts
        dff = dfo.apply(lambda x: pd.factorize(x)[0])
        dff = IdenticalFilter(self._threshold).select(dff)
        # select columns to be dropped
        to_drop = [c for c in dfo_columns if c not in dff.columns]
        return df.drop(to_drop, axis=1)


class UniqueFilter:
    """Remove columns containing only categorical, unique elements."""

    def __init__(self, threshold: float = 0.5):
        """

        Args:
            threshold: If this percentage of values is unique,
                filter the column.

        """
        self._threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        uniques = list()
        for column in df:
            if df[column].dtype.name in ["object", "category"]:
                unique = df[column].nunique() / df[column].count()
                if unique > self._threshold:
                    uniques.append(column)
        return df.drop(uniques, axis=1)


class CorrelationFilter:
    """Train decision stump for every feature individually."""

    def __init__(self, threshold: float = 0.95, data_size: int = 1000):
        """

        Args:
            threshold: Features that make the same prediction for
                `threshold` percentage of rows are discarded.
            data_size: Number of examples to sample.

        """
        self._threshold = threshold
        self._data_size = data_size

    def predictions(self, df: pd.DataFrame, target: Label = None) -> pd.DataFrame:
        """Make predictions.

        Returns:
            Dataframe with same shape as `df` containing predictions for
            every feature.

        """
        # sample DF
        # if len(df.index) > self._data_size:
        #     df = df.sample(self._data_size)
        # prepare data for mercs
        data, nominal = to_mercs(df)
        data = data.values
        # # compute stratification target and size
        # if target and not is_numeric_dtype(df[target]):
        #     target_df = df[[target]]
        #     target_size = target_df[target].value_counts()[-1]
        # else:
        #     target_df = None
        #     target_size = 0.5
        # make train/test split
        train, test = train_test_split(data, test_size=0.5)
        test = np.nan_to_num(test)
        # mask
        m_code = np.array([[0, 1]])
        # target index
        target_i = df.columns.get_loc(target)
        # perform predictions
        predictions = pd.DataFrame()
        for i, column in enumerate(df.columns):
            if column == target:
                continue
            model = Mercs(classifier_algorithm="DT", max_depth=4, min_leaf_node=10)
            model.fit(
                train[:, [i, target_i]], nominal_attributes=nominal, m_codes=m_code
            )
            predictions[column] = model.predict(
                test[:, [i, target_i]], q_code=m_code[0]
            )
        return predictions

    def select(self, df: pd.DataFrame, target) -> pd.DataFrame:
        """Perform selection.

        Returns:
            A dataframe containing only selected features.

        """
        # get predictions
        predictions = self.predictions(df, target)
        predictions = predictions.loc[:, predictions.apply(pd.Series.nunique) != 1]
        # get columns similar to another columns
        similar = set()
        for i, ci in enumerate(predictions.columns):
            if ci in similar:
                break
            column = predictions[ci]
            column_type = is_numeric_dtype(df[ci])
            for cj in predictions.columns[i + 1 :]:
                if cj in similar:
                    break
                other = predictions[cj]
                other_type = is_numeric_dtype(df[cj])
                if (column_type and other_type) or (not column_type and not other_type):
                    d = np.count_nonzero(column == other) / len(column)
                    if d > self._threshold:
                        print("{} is too similar to {}".format(cj, ci))
                        similar.add(cj)
        # similar ones and return
        return df.drop(similar, axis=1)


default_pruner = StackedFilter(
    [MissingFilter(), GreedyMissingFilter(), ConstantFilter(), IdenticalFilter()]
)
default_filter = StackedFilter([BijectiveFilter(), UniqueFilter()])