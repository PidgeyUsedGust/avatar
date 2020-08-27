import numpy as np
import pandas as pd
from abc import abstractmethod, ABC
from pandas._typing import Label
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from mercs.core import Mercs
from .utilities import to_mercs


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


class ConstantFilter:
    """Remove columns with constants."""

    def select(self, df: pd.DataFrame, target=None):
        return df.loc[:, (df != df.iloc[0]).any()]


class IdenticalFilter:
    """Remove identical columns.
    
    Remove numerical columns that are identical.

    """

    def select(self, df: pd.DataFrame, target=None):
        return df.drop(self.duplicates(df), axis=1)

    def duplicates(self, df: pd.DataFrame):
        duplicates = set()
        for i in range(df.shape[1]):
            col_one = df.iloc[:, i]
            for j in range(i + 1, df.shape[1]):
                col_two = df.iloc[:, j]
                if col_one.equals(col_two):
                    duplicates.add(df.columns[j])
        return duplicates


class BijectiveFilter:
    """Remove categorical columns that are a bijection."""

    def select(self, df: pd.DataFrame, target=None):
        return df.drop(self.duplicates(df), axis=1)

    def duplicates(self, df: pd.DataFrame):
        # select only object
        df = df.select_dtypes(include="object")
        duplicates = set()
        for i in range(df.shape[1]):
            col_one = df.iloc[:, i]
            for j in range(i + 1, df.shape[1]):
                col_two = df.iloc[:, j]
                if (col_one.factorize()[0] == col_two.factorize()[0]).all():
                    duplicates.add(df.columns[j])
        return duplicates


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


# class CorrelationFilter:
#     """Train decision stump for every feature individually."""

#     def __init__(self, threshold: float = 0.95):
#         """

#         Args:
#             threshold: Features that make the same prediction for
#                 `threshold` percentage of rows are discarded.

#         """
#         self.threshold = threshold

#     def predictions(self, df: pd.DataFrame, target: Label = None) -> pd.DataFrame:
#         """Make predictions.

#         Returns:
#             Dataframe with same shape as `df` containing predictions for
#             every feature.

#         """

#         # prepare data for mercs
#         data, nominal = to_mercs(df)
#         data = data.values
#         # data_test = np.nan_to_num(data)

#         if target:
#             target_df = df[[target]]
#             target_size = target_df[target].value_counts()[-1]
#         else:
#             target_column = None
#             target_size = 0.5

#         train, test = train_test_split(data, test_size=target_size, stratify=target_df)
#         test = np.nan_to_num(test)

#         # initialise mask
#         base_m_code = to_m_codes(df.columns, target)
#         base_m_code[base_m_code == 0] = -1

#         # perform predictions
#         predictions = pd.DataFrame(columns=df.columns)
#         for i, column in enumerate(df.columns):
#             if column == target:
#                 continue
#             m_code = np.copy(base_m_code)
#             m_code[:, i] = 0
#             model = Mercs(classifier_algorithm="DT", max_depth=1)
#             model.fit(train, nominal_attributes=nominal, m_codes=m_code)
#             # print(column)
#             # print(export_text(model.m_list[0].model))
#             predictions[column] = model.predict(test, q_code=m_code[0])
#         return predictions

#     def select(self, df: pd.DataFrame, target) -> pd.DataFrame:
#         """Perform selection.

#         Returns:
#             A dataframe containing only selected features.

#         """

#         # get predictions
#         predictions = self.predictions(df, target)

#         # get columns similar to another columns
#         similar = set()
#         for i, ci in enumerate(predictions.columns):
#             if ci in similar:
#                 break
#             column = predictions[ci]
#             column_type = is_numeric_dtype(df[ci])
#             for cj in predictions.columns[i + 1 :]:
#                 if cj in similar:
#                     break
#                 other = predictions[cj]
#                 other_type = is_numeric_dtype(df[cj])
#                 if (column_type and other_type) or (not column_type and not other_type):
#                     # print(ci, cj, np.count_nonzero(column != other))
#                     d = np.count_nonzero(column != other) / len(column)
#                     if d > self.threshold:
#                         print("{} is too similar to {}".format(cj, ci))
#                         similar.add(cj)

#         # similar ones and return
#         return df.drop(similar, axis=1)


class CorrelationFilter:
    """Train decision stump for every feature individually."""

    def __init__(self, threshold: float = 0.95):
        """
        
        Args:
            threshold: Features that make the same prediction for
                `threshold` percentage of rows are discarded.

        """
        self.threshold = threshold

    def predictions(self, df: pd.DataFrame, target: Label = None) -> pd.DataFrame:
        """Make predictions.
        
        Returns:
            Dataframe with same shape as `df` containing predictions for
            every feature.
            
        """
        # prepare data for mercs
        data, nominal = to_mercs(df)
        data = data.values
        # compute stratification target and size
        if target:
            target_df = df[[target]]
            target_size = target_df[target].value_counts()[-1]
        else:
            target_df = None
            target_size = 0.5
        # make train/test split
        train, test = train_test_split(data, test_size=target_size, stratify=target_df)
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
            model = Mercs(classifier_algorithm="DT", max_depth=4, min_samples_leaf=10)
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
                    if d > self.threshold:
                        # print("{} is too similar to {}".format(cj, ci))
                        similar.add(cj)
        # similar ones and return
        return df.drop(similar, axis=1)
