import pandas as pd
import numpy as np
from abc import abstractmethod, ABC
from typing import Hashable
from pandas.core.algorithms import value_counts
from sklearn.metrics import adjusted_mutual_info_score as ami
from scipy.stats import kruskal, kendalltau, ks_2samp
from scipy.stats.contingency import crosstab, chi2_contingency
from statsmodels.stats.multitest import multipletests


class Filter:
    """Generic filter."""

    def select(self, df: pd.DataFrame, target: Hashable = None) -> pd.DataFrame:
        return df


class StackedFilter(Filter):
    """Combine differenct filters."""

    def __init__(self, selectors):
        self._selectors = selectors

    def select(self, df: pd.DataFrame, target: Hashable = None):
        for selector in self._selectors:
            df = selector.select(df, target=target)
        return df


class MissingFilter(Filter):
    """Remove columns missing at least a percentage of values."""

    def __init__(self, threshold: float = 0.5):
        """

        Args:
            threshold: Remove columns that have this much
                missing values.

        """
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        return df.dropna(axis=1, thresh=self.threshold * len(df.index))


# class GreedyMissingFilter:
#     """Greedily remove column with most missing values."""

#     def __init__(self, min_full: int = 10):
#         self._min_full = min_full

#     def select(self, df: pd.DataFrame, target: Hashable = None):
#         m_count = df.isna().sum(axis=0)
#         missing = m_count.index[(m_count.values).argsort()].tolist()
#         while df.notna().all(axis=1).sum() < self._min_full:
#             remove = missing.pop()
#             df = df.drop(remove, axis=1)
#         return df


class ConstantFilter(Filter):
    """Remove columns with constants."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        # quickly remove all unique
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]
        if self.threshold == 1:
            return df
        # go over columns
        to_drop = list()
        for column in df:
            counts = df[column].value_counts(normalize=True)
            if len(counts) == 0 or counts.iloc[0] > self.threshold:
                to_drop.append(column)
        return df.drop(to_drop, axis=1)


class InfinityFilter(Filter):
    """Remove columns with infinity."""

    def select(self, df: pd.DataFrame, target: Hashable = None) -> pd.DataFrame:
        to_drop = list()
        for column in df.select_dtypes("float64"):
            if np.isinf(df[column]).any():
                print(column)
                to_drop.append(column)
        return df.drop(to_drop, axis=1)


class IdenticalFilter(Filter):
    """Remove identical columns.

    Remove all columns that are basically identical.

    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        df = df.loc[:, ~df.T.duplicated(keep="first")]
        if self.threshold == 1:
            return df
        todrop = list()
        for dtype in df.dtypes.unique():
            dtypef = df.select_dtypes(dtype)
            for column in dtypef.columns.tolist():
                if column not in df:
                    continue
                i = dtypef.columns.get_loc(column)
                col_one = dtypef.iloc[:, i]
                compare = dtypef.iloc[:, i + 1 :]
        return df.drop(todrop, axis="columns")


class BijectiveFilter:
    """Remove categorical columns that are (almost) bijective."""

    def __init__(self, threshold: float = 1.0):
        self._threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        # get all objects
        dfo = df.select_dtypes(include="string")
        dfo_columns = dfo.columns
        # factorize whole object dataframe and remove exacts
        dff = dfo.apply(lambda x: pd.factorize(x)[0])
        dff = IdenticalFilter(self._threshold).select(dff)
        # select columns to be dropped
        to_drop = [c for c in dfo_columns if c not in dff.columns]
        return df.drop(to_drop, axis=1)


class MutualInformationFilter:
    """Remove categorical columns based on AMI."""

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold

    def select(self, df: pd.DataFrame, target=None):
        # get all objects
        dfo = df.select_dtypes(include=["string", "category"])
        # factorize whole object dataframe and
        # remove exact duplicates
        dff = dfo.apply(lambda x: pd.factorize(x)[0])
        dff = IdenticalFilter(self.threshold).select(dff)
        if self.threshold == 1:
            return dff
        # need to remove approximate ones as well
        to_drop = list()
        for i, c1 in enumerate(dff.columns):
            if c1 in to_drop:
                continue
            for c2 in dff.columns[i + 1 :]:
                if ami(dff[c1], dff[c2]) > self.threshold:
                    to_drop.append(c2)
        return df.drop(to_drop, axis=1)


class StringFilter(Filter):
    """Filter strings.

    Implement as filter to fit in with the rest of the
    framework. All remaining columns are encoded in place
    and used for selection.

    """

    def select(self, df: pd.DataFrame, target: Hashable) -> pd.DataFrame:
        return df.select_dtypes(exclude=["string", "object"])


# class UniqueFilter(Filter):
#     """Remove columns containing only categorical, unique elements."""

#     def __init__(self, threshold: float = 0.5):
#         """

#         Args:
#             threshold: If this percentage of values is unique,
#                 filter the column.

#         """
#         self._threshold = threshold

#     def select(self, df: pd.DataFrame, target=None):
#         uniques = list()
#         for column in df:
#             if df[column].dtype.name in ["string"]:
#                 unique = df[column].nunique() / df[column].count()
#                 if unique > self._threshold:
#                     uniques.append(column)
#         return df.drop(uniques, axis=1)


class FreshFilter(Filter):
    """Use the FRESH algorithm.

    See [1]_ for more an introduction. As oppused to
    the original work, we have four different cases.

    * bool -> real
    * real -> real
    * bool -> cat
    * real -> cat

    .. [1] https://tsfresh.readthedocs.io/en/latest/text/feature_filtering.html

    """

    def __init__(self, q: float = 0.1) -> None:
        super().__init__()
        self._q = q

    def select(self, df: pd.DataFrame, target: Hashable) -> pd.DataFrame:
        """Perform FRESH filter."""

        y = df[target]
        i = df.columns.get_loc(target)
        category = y.dtype == "category"

        # generate list of p values
        p_values = list()
        for name, x in df.items():
            # skip the target
            if name == target:
                continue
            # boolean feature
            if x.dtype == "category":
                if category:
                    p_values.append(FreshFilter._test_bool_cat(x, y))
                else:
                    p_values.append(FreshFilter._test_bool_real(x, y))
            # numerical feature
            else:
                if category:
                    p_values.append(FreshFilter._test_real_cat(x, y))
                else:
                    p_values.append(FreshFilter._test_real_real(x, y))

        # run Benjamini/Yekutieli procedure
        reject, _, _, _ = multipletests(p_values, self._q, method="fdr_by")

        # insert target
        reject = np.insert(reject, i, False)

        return df[df.columns[~reject]]

    @staticmethod
    def _test_bool_real(c1: pd.Series, c2: pd.Series) -> float:
        """"""
        values = np.unique(c2)
        groups = [c1[c2 == v] for v in values]
        _, p = ks_2samp(*groups)
        return p

    @staticmethod
    def _test_real_real(c1: pd.Series, c2: pd.Series) -> float:
        """Significance of real to real feature.

        Similar to FRESH, we use the Kendall tau test.

        Args:
            c1, c2: Two real series.

        Returns:
            p-value of the significance test.

        """
        _, p = kendalltau(c1, c2)
        return p

    @staticmethod
    def _test_bool_cat(c1: pd.Series, c2: pd.Series) -> float:
        """"""
        _, table = crosstab(c1, c2)
        _, p, _, _ = chi2_contingency(table)
        return p

    @staticmethod
    def _test_real_cat(c1: pd.Series, c2: pd.Series) -> float:
        """

        Uses the Kruskal–Wallis H test if there are more than
        two categories, else use the Kolmogorov-Smirnov test (as
        in tsfresh).

        Args:
            c1: Real feature.
            c2: Categorical target (can also be boolean).

        Returns:
            The p-value of the feature significance test.

        """

        values = np.unique(c2)
        groups = [c1[c2 == v] for v in values]

        if len(groups) == 2:
            _, p = ks_2samp(*groups)
        else:
            _, p = kruskal(*groups)

        return p


# class CorrelationFilter:
#     """Train decision stump for every feature individually."""

#     def __init__(self, threshold: float = 0.95, data_size: int = 1000):
#         """

#         Args:
#             threshold: Features that make the same prediction for
#                 `threshold` percentage of rows are discarded.
#             data_size: Number of examples to sample.

#         """
#         self._threshold = threshold
#         self._data_size = data_size

#     def predictions(self, df: pd.DataFrame, target: Hashable = None) -> pd.DataFrame:
#         """Make predictions.

#         Returns:
#             Dataframe with same shape as `df` containing predictions for
#             every feature.

#         """
#         # sample DF
#         # if len(df.index) > self._data_size:
#         #     df = df.sample(self._data_size)
#         # prepare data for mercs
#         data, nominal = to_mercs(df)
#         data = data.values
#         # # compute stratification target and size
#         # if target and not is_numeric_dtype(df[target]):
#         #     target_df = df[[target]]
#         #     target_size = target_df[target].value_counts()[-1]
#         # else:
#         #     target_df = None
#         #     target_size = 0.5
#         # make train/test split
#         train, test = train_test_split(data, test_size=0.5)
#         test = np.nan_to_num(test)
#         # mask
#         m_code = np.array([[0, 1]])
#         # target index
#         target_i = df.columns.get_loc(target)
#         # perform predictions
#         predictions = pd.DataFrame()
#         for i, column in enumerate(df.columns):
#             if column == target:
#                 continue
#             model = Mercs(classifier_algorithm="DT", max_depth=4, min_leaf_node=10)
#             model.fit(
#                 train[:, [i, target_i]], nominal_attributes=nominal, m_codes=m_code
#             )
#             predictions[column] = model.predict(
#                 test[:, [i, target_i]], q_code=m_code[0]
#             )
#         return predictions

#     def select(self, df: pd.DataFrame, target) -> pd.DataFrame:
#         """Perform selection.

#         Returns:
#             A dataframe containing only selected features.

#         """
#         # get predictions
#         predictions = self.predictions(df, target)
#         predictions = predictions.loc[:, predictions.apply(pd.Series.nunique) != 1]
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
#                     d = np.count_nonzero(column == other) / len(column)
#                     if d > self._threshold:
#                         print("{} is too similar to {}".format(cj, ci))
#                         similar.add(cj)
#         # similar ones and return
#         return df.drop(similar, axis=1)


default_pruner = StackedFilter(
    [
        MissingFilter(),
        ConstantFilter(),
        IdenticalFilter(),
    ]
)
default_filter = BijectiveFilter()
