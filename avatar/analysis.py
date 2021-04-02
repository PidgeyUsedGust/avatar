"""Analyse quality of a dataset by machine learning.

We use MERCS as a backend as this allows us to have targeted
and targetless wrangling in one clean framework.

"""
import warnings
import random
import pandas as pd
import numpy as np
from pandas._typing import Label
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Set, Optional, Union
from functools import cached_property
from collections import defaultdict
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import export_text
# from mercs.core import Mercs
# from mercs.algo.selection import base_selection_algorithm, random_selection_algorithm
from .utilities import normalize, to_mercs, to_m_codes


class FeatureEvaluator:
    """Feature evaluator.

    Given a mask of which features to include, get
    accuracy of the model and relevance of individual
    features.

    """

    def __init__(
        self,
        n_folds: int = 4,
        n_samples: Union[int, float] = 1.0,
        test_size: Union[int, float] = 0.2,
        **configuration
    ):
        """

        Args:
            n_folds: Number of folds.
            n_samples: Number of examples to sample from the dataframe. If
                an integer, use that many examples. If a float, use that
                percentage of examples. By default, use the whole dataset.
            method: Method to use for computing feature relevances from
                decision trees. Can be `shap` (default) or `None`.

        Kwargs:
            Any other arguments are passed to the `Mercs` constructor.

        """
        self._n_folds = n_folds
        self._n_samples = n_samples
        self._test_size = test_size
        # settings for training
        self._m_args = dict(max_depth=4)
        self._m_args.update(configuration)
        # cache for model
        self._cache = dict()

    def fit(self, df: pd.DataFrame, target: Label = None):
        self._columns = df.columns
        self._target = target
        self._target_index = df.columns.get_loc(target)
        # get nominal columns, but don't store data
        # as that will be generated in the folds.
        _, nominal = to_mercs(df)
        self._nominal = nominal
        # make split
        self._folds = list()
        for _ in range(self._n_folds):
            train, test = self._split(df)
            train = train.values
            test = np.nan_to_num(test.values)
            self._folds.append((train, test))

    def predict(self, mask: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
        """Perform predictions for all folds.

        Returns:
            A list of (fold, code, prediction) tuples.

        """
        predict = list()
        models = self._models(mask)
        for i, model in enumerate(models):
            _, test = self._folds[i]
            for m_code in model.m_codes:
                prediction = model.predict(test, q_code=m_code)
                predict.append((i, m_code, prediction))
        return predict

    def accuracy(self, mask: np.ndarray) -> float:
        """Average accuracy over all folds."""
        accuracies = [
            self._score(code, fold, prediction)
            for fold, code, prediction in self.predict(mask)
        ]
        return np.mean(accuracies)

    def importances(self, mask: np.ndarray) -> np.ndarray:
        importances = np.zeros((len(self._folds), len(self._columns)))
        for i, model in enumerate(self._models(mask)):
            importances[i] = np.sum(model.nrm_shaps, axis=0)
        return np.mean(importances, axis=0)

    def evaluate(self, mask: np.ndarray):
        """Evaluate.

        Args:
            mask: Mask of which features to use.

        Returns:
            Tuple of accuracy and individual feature importances.

        """
        return self.accuracy(mask), self.importances(mask)

    def _models(self, mask: np.ndarray) -> List["Mercs"]:
        """Generate models for a given mask.

        Returns:
            A model train/test split.

        """
        key = mask.tobytes()
        if key not in self._cache:
            self._cache[key] = list()
            code = self._code(mask)
            for (train, _) in self._folds:
                model = Mercs(**self._m_args)
                model.fit(train, nominal_attributes=self._nominal, m_codes=code)
                # compute SHAP values
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.avatar(train, keep_abs_shap=True, check_additivity=False)
                self._cache[key].append(model)
        return self._cache[key]

    def _code(self, mask: np.ndarray) -> np.ndarray:
        """Generate m_codes for a given feature mask."""
        mask = mask.astype(bool)
        # no target, use mercs selection algorithm
        if self._target is None:
            m_code = random_selection_algorithm(
                self._metadata(), nb_targets=1, nb_iterations=1, fraction_missing=0.0
            )
        # else make m_code for target
        else:
            m_code = to_m_codes(self._columns, self._target)
        # hide mask
        output = m_code == 1
        m_code[:, ~mask] = -1
        m_code[output] = 1
        return m_code

    def _query(self, code: np.ndarray) -> np.ndarray:
        """Convert m_code back into query."""
        code = np.copy(code)
        code[code == -1] = 0
        return code

    def _score(self, code: np.ndarray, fold: int, prediction: np.ndarray) -> float:
        """Compute score of prediction.

        For classification, use accuracy.
        For regression, use mean squared error.

        """
        truth = self._truth(code, fold)
        # classification
        if self._target_index in self._nominal:
            return accuracy_score(truth, prediction)
        # or regression
        else:
            return max(r2_score(truth, prediction), 0)
            # return mean_squared_error(truth, prediction)

    def _truth(self, code: np.ndarray, fold: int) -> np.ndarray:
        """Get truth for m_code."""
        return self._folds[fold][1][:, code == 1][:, 0]

    def _metadata(self):
        """Return metadata."""
        attributes = set(range(len(self._columns)))
        return dict(
            n_attributes=len(attributes),
            nominal_attributes=self._nominal,
            numeric_attributes=attributes - self._nominal,
        )

    def _split(self, df: pd.DataFrame, min_full: int = 10):
        """Make train/test split.

        Custom train/test split tha guarantees that the
        training data contains sufficient rows without nan.

        Each fold is shuffled to minimize the probability
        of arbitrary encodings becoming good features.

        Args:
            min_full: Minimal number of rows without missing values.

        """

        # compute data size
        if isinstance(self._n_samples, float):
            n_samples = int(self._n_samples * len(df.index))
        else:
            n_samples = min(len(df.index), self._n_samples)

        # compute test size
        if isinstance(self._test_size, float):
            test_size = int(self._test_size * n_samples)

        # shuffle rows and convert to mercs.
        df, _ = to_mercs(df.sample(frac=1, random_state=1337))

        # get min_full rows from the dataframe
        full = df[df.notna().all(axis=1)].sample(min_full, random_state=1337)
        rest = df[~df.index.isin(full.index)]

        # sample test and train
        test = rest.sample(test_size, random_state=1337)
        train = rest[~rest.index.isin(test.index)].sample(
            n_samples - test_size - min_full, random_state=1337
        )

        return pd.concat((full, train), axis=0), test

    def __str__(self) -> str:
        return "FeatureEvaluator(folds={}, max_depth={}, sample={})".format(
            self._n_folds, self._m_args["max_depth"], self._n_samples
        )


class MercsFeatureEvaluator(FeatureEvaluator):
    """

    Rather than learning models at runtime, they are learned
    when fitting. At runtime, queries are generated based on the
    given mask.

    """

    def __init__(
        self,
        n_folds: int = 5,
        n_samples: Union[int, float] = 1.0,
        test_size: Union[int, float] = 0.2,
        **configuration
    ):
        """

        Kwargs:
            Are passed to the Mercs model. This can be used to set
            features like `fraction_missing` and `nb_iterations`.

        """
        self._n_folds = n_folds
        self._n_samples = n_samples
        self._test_size = test_size
        # settings for training
        self._m_args = dict(
            max_depth=4,
            selection_algorithm="random",
            nb_targets=1,
            nb_iterations=10,
            fraction_missing="sqrt",
            prediction_algorithm="rw",
            max_steps=8,
            nb_walks=5,
        )
        self._m_args.update(configuration)
        # cache for model
        self._models = list()
        self._folds = list()

    def fit(self, df: pd.DataFrame, target: Label = None):
        self._columns = df.columns
        self._target = target
        self._target_index = df.columns.get_loc(target)
        # get nominal columns, but don't store data
        # as that will be generated in the folds.
        _, nominal = to_mercs(df)
        self._nominal = nominal
        # make splits
        self._models = list()
        self._folds = list()
        for _ in range(self._n_folds):
            # generate fold
            train, test = self._split(df)
            train = train.values
            test = np.nan_to_num(test.values)
            self._folds.append((train, test))
            # train model
            model = Mercs(**self._m_args)
            model.fit(train, nominal_attributes=self._nominal)
            self._models.append(model)

    def __str__(self) -> str:
        return "MercsFeatureEvaluator(max_depth={}, sample={})".format(
            self._m_args["max_depth"], self._n_samples
        )


class DatasetEvaluator:
    """Dataset evaluator."""

    def __init__(self, n_folds: int = 5, **configuration):
        self._n_folds = n_folds
        self._m_args = dict()
        self._m_args.update(configuration)

    def fit(self, df: pd.DataFrame, target: Optional[Label] = None):
        # prepare data
        data, nominal = to_mercs(df)
        self._data = data.values
        self._nominal = nominal
        self._columns = df.columns
        self._target = target
        self._target_index = df.columns.get_loc(target)

    def evaluate(self, get_importances=False):
        """Evaluate on folds. """
        # generate a code
        code = self._code()
        # run the train, test splits.
        accuracies = list()
        importances = list()
        for train, test in self._folds():
            test = np.nan_to_num(test)
            # learn model
            model = Mercs(**self._m_args)
            model.fit(train, nominal_attributes=self._nominal, m_codes=code)
            if get_importances:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.avatar(train, keep_abs_shap=True, check_additivity=False)
                    importances.append(np.sum(model.nrm_shaps, axis=0))
            # make prediction for each model
            for m_code in model.m_codes:
                prediction = model.predict(test, q_code=m_code)
                truth = test[:, m_code == 1][:, 0]
                # classification
                if self._target_index in self._nominal:
                    accuracy = accuracy_score(truth, prediction)
                # or regression
                else:
                    accuracy = mean_squared_error(truth, prediction)
                accuracies.append(accuracy)
        if get_importances:
            return np.mean(accuracies), np.mean(importances, axis=0)
        return np.mean(accuracies)

    def _code(self) -> np.ndarray:
        """Generate m_codes."""
        # no target, use mercs selection algorithm
        if self._target is None:
            m_code = random_selection_algorithm(
                self._metadata(), nb_targets=1, nb_iterations=1, fraction_missing=0.0
            )
        # else make m_code for target
        else:
            m_code = to_m_codes(self._columns, self._target)
        return m_code

    def _folds(self):
        """Make train/test splits.

        Args:
            min_full: Minimal number of rows without missing values.

        Yields:
            A (train, test) split at each iteration.

        """
        for train, test in KFold(
            n_splits=self._n_folds, shuffle=True, random_state=1337
        ).split(self._data):
            yield self._data[train], self._data[test]

    def _metadata(self):
        """Return metadata."""
        attributes = set(range(len(self._columns)))
        return dict(
            n_attributes=len(attributes),
            nominal_attributes=self._nominal,
            numeric_attributes=attributes - self._nominal,
        )


# class ColumnSampler:
#     """Column sampler."""

#     def __init__(self, df: pd.DataFrame):
#         self._df = df
#         self._df_notna = df.notna()

#     @abstractmethod
#     def sample(self) -> pd.DataFrame:
#         pass

#     @cached_property
#     def to_sample(self) -> List[Label]:
#         """Columns that need sampling."""
#         return self._df_notna.columns[~self._df_notna.all()].to_list()

#     @cached_property
#     def not_to_sample(self) -> List[Label]:
#         """Columns that don't need sampling."""
#         return self._df_notna.columns[self._df_notna.all()].to_list()

#     def is_valid(self, df: pd.DataFrame):
#         """Check if set of columns is valid."""
#         return df.all(axis=1).sum() > 10

#     def make_result(self, sampled: List[Label]) -> pd.DataFrame:
#         """Get result."""
#         return pd.concat((self._df[self.not_to_sample], self._df[sampled]), axis=1)


# class UniformColumnSampler(ColumnSampler):
#     """Sample uniformly."""

#     def __init__(self, df: pd.DataFrame):
#         super().__init__(df)

#     def sample(self) -> pd.DataFrame:
#         """Sample a set of columns with a full row.

#         Iteratively remove a column until valid solution found.

#         """
#         df = self._df_notna[self.to_sample]
#         while len(df.columns) > 0:
#             if self.is_valid(df):
#                 break
#             rm = random.choice(df.columns)
#             df = df.drop(rm, axis=1)
#         return self.make_result(df.columns)


# class WeightedColumnSampler(ColumnSampler):
#     """Weigh samples by number of nans."""

#     def __init__(self, df: pd.DataFrame):
#         super().__init__(df)

#     def sample(self) -> pd.DataFrame:
#         df = self._df_notna[self.to_sample]
#         ws = self.weights[self.to_sample]
#         while len(df.columns) > 0:
#             if self.is_valid(df):
#                 break
#             rm = random.choices(df.columns, weights=ws, k=1)
#             df = df.drop(rm, axis=1)
#             ws = ws.drop(rm)
#         return self.make_result(df.columns)

#     @cached_property
#     def weights(self) -> pd.Series:
#         weights = (~self._df_notna).sum(axis=0)
#         weights[weights != 0] = np.log(weights[weights != 0])
#         weights = normalize(weights)
#         return weights


# class SmartColumnSampler(WeightedColumnSampler):
#     """Weighs by how often picked before."""

#     def __init__(self, df: pd.DataFrame):
#         super().__init__(df)
#         self._counts = pd.Series(0, index=df.columns)

#     def sample(self) -> pd.DataFrame:
#         df = self._df_notna[self.to_sample]
#         ws = self.weights[self.to_sample]
#         while len(df.columns) > 0:
#             if self.is_valid(df):
#                 break
#             rm = random.choices(df.columns, weights=ws, k=1)
#             df = df.drop(rm, axis=1)
#             ws = ws.drop(rm)
#         return self.make_result(df.columns)

#     @property
#     def counts(self):
#         """Normalized counts."""
#         return normalize(self._counts)

#     @property
#     def smart_weights(self):
#         """Combine weights with count."""
#         return self.counts.add(super().weights) / 2
