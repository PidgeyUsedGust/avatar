"""Analyse quality of a dataset by machine learning.

We use MERCS as a backend as this allows us to have targeted
and targetless wrangling in one clean framework.

"""
import random
import pandas as pd
import numpy as np
from pandas._typing import Label
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Set, Optional, Union
from functools import cached_property
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import export_text
from mercs.core import Mercs
from mercs.algo.selection import base_selection_algorithm, random_selection_algorithm
from .utilities import normalize, to_mercs, to_m_codes


# class FeatureEvaluator:
#     """Feature evaluator.

#     Given a mask of which features to include, get
#     accuracy of the model and relevance of individual
#     features.

#     """

#     def __init__(
#         self,
#         method: Optional[str] = None,
#         n_samples: Union[int, float] = 1.0,
#         test_size: Union[int, float] = 0.1,
#         **configuration
#     ):
#         """

#         Args:
#             n_samples: Number of examples to sample from the dataframe. If
#                 an integer, use that many examples. If a float, use that
#                 percentage of examples. By default, use the whole dataset.
#             test_size: Number of examples to use for testing. If an integer,
#                 use that number of examples. If a float, use that percentage
#                 of examples.
#             method: Method to use for computing feature relevances from
#                 decision trees. Can be `shap` (default) or `None`.

#         Kwargs:
#             Any other arguments are passed to the `Mercs` constructor.

#         """
#         self._n_samples = n_samples
#         self._test_size = test_size
#         # settings for training
#         self._m_args = dict(calculation_method_feature_importances=method)
#         self._m_args.update(configuration)
#         # cache for models
#         self._cache = dict()

#     def fit(self, df: pd.DataFrame, target: Label = None):
#         # compute data size
#         if isinstance(self._n_samples, float):
#             self._n_samples = int(self._n_samples * len(df.index))
#         # compute test size
#         if isinstance(self._test_size, float):
#             self._test_size = int(self._test_size * self._n_samples)
#         # convert to mercs
#         data, nominal = to_mercs(df)
#         self._columns = df.columns
#         self._target = target
#         self._nominal = nominal
#         # make split
#         train, test = self._split(data)
#         self._train = train.values
#         self._test = np.nan_to_num(test.values)

#     def predict(self, mask: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
#         """Perform predictions.

#         Returns:
#             A list codes and the predictions for that code on the test set.

#         """
#         predict = list()
#         model = self.model(mask)
#         for m_code in model.m_codes:
#             prediction = model.predict(self._test, q_code=m_code)
#             predict.append((m_code, prediction))
#         return predict

#     def accuracy(self, mask: np.ndarray) -> float:
#         accuracies = list()
#         for code, prediction in self.predict(mask):
#             accuracies.append(accuracy_score(self._truth(code), prediction))
#         return np.mean(accuracies)

#     def importances(self, mask: np.ndarray) -> np.ndarray:
#         return np.sum(self.model(mask).m_fimps, axis=0)

#     def evaluate(self, mask: np.ndarray):
#         """Evaluate.

#         Args:
#             mask: Mask of which features to use.

#         Returns:
#             Tuple of accuracy and individual feature importances.

#         """
#         return self.accuracy(mask), self.importances(mask)

#     def model(self, mask: np.ndarray) -> Mercs:
#         """Generate model for a given mask.

#         Returns:
#             A model for every code in `self._code(mask)`.

#         """
#         key = mask.tobytes()
#         if key not in self._cache:
#             model = Mercs(**self._m_args)
#             model.fit(
#                 self._train, nominal_attributes=self._nominal, m_codes=self._code(mask)
#             )
#             self._cache[key] = model
#             # print(mask)
#             # print([self._columns[i] for i, v in enumerate(mask) if v == 0])
#             # print(export_text(model.m_list[0].model))
#         return self._cache[key]

#     def _code(self, mask: np.ndarray) -> np.ndarray:
#         """Generate m_codes for a given feature mask."""

#         mask = mask.astype(bool)

#         # no target, use mercs selection algorithm
#         if self._target is None:
#             m_code = random_selection_algorithm(
#                 self._metadata(), nb_targets=1, nb_iterations=1, fraction_missing=0.0
#             )

#         # else make m_code for target
#         else:
#             m_code = to_m_codes(self._columns, self._target)

#         # hide mask
#         output = m_code == 1
#         m_code[:, mask] = -1
#         m_code[output] = 1

#         return m_code

#     def _query(self, code: np.ndarray) -> np.ndarray:
#         """Convert m_code back into query."""
#         code = np.copy(code)
#         code[code == -1] = 0
#         return code

#     def _truth(self, code: np.ndarray) -> np.ndarray:
#         """Get truth for m_code."""
#         return self._test[:, code == 1][:, 0]

#     def _metadata(self):
#         """Return metadata."""
#         attributes = set(range(len(self._columns)))
#         return dict(
#             n_attributes=len(attributes),
#             nominal_attributes=self._nominal,
#             numeric_attributes=attributes - self._nominal,
#         )

#     def _split(self, df: pd.DataFrame, min_full: int = 10):
#         """Make train/test split.

#         Custom train/test split tha guarantees that the
#         training data contains sufficient rows without nan.

#         Args:
#             min_full: Minimal number of rows without missing values.

#         """

#         # get min_full rows from the dataframe
#         full = df[df.notna().all(axis=1)].sample(min_full)
#         rest = df[~df.index.isin(full.index)]

#         # sample test and train
#         test = rest.sample(self._test_size)
#         train = rest[~rest.index.isin(test.index)].sample(
#             self._n_samples - self._test_size - min_full
#         )

#         return pd.concat((full, train), axis=0), test


class FeatureEvaluator:
    """Feature evaluator.
    
    Given a mask of which features to include, get
    accuracy of the model and relevance of individual
    features.
    
    """

    def __init__(
        self,
        n_folds: int = 5,
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
        self._m_args = dict()
        self._m_args.update(configuration)
        # cache for model
        self._cache = dict()

    def fit(self, df: pd.DataFrame, target: Label = None):
        # compute data size
        if isinstance(self._n_samples, float):
            self._n_samples = int(self._n_samples * len(df.index))
        # compute test size
        if isinstance(self._test_size, float):
            self._test_size = int(self._test_size * self._n_samples)
        self._columns = df.columns
        self._target = target
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
        models = self.models(mask)
        for i, model in enumerate(models):
            _, test = self._folds[i]
            for m_code in model.m_codes:
                prediction = model.predict(test, q_code=m_code)
                predict.append((i, m_code, prediction))
        return predict

    def accuracy(self, mask: np.ndarray) -> float:
        """Average accuracy over all folds."""
        accuracies = [
            accuracy_score(self._truth(code, fold), prediction)
            for fold, code, prediction in self.predict(mask)
        ]
        return np.mean(accuracies)

    def importances(self, mask: np.ndarray) -> np.ndarray:
        importances = np.zeros((len(self._folds), len(self._columns)))
        for i, model in enumerate(self.models(mask)):
            importances[i] = np.sum(model.m_fimps, axis=0)
            # importances[i] = np.sum(model.nrm_shaps, axis=0)
        return np.mean(importances, axis=0)

    def evaluate(self, mask: np.ndarray):
        """Evaluate.
        
        Args:
            mask: Mask of which features to use.
        
        Returns:
            Tuple of accuracy and individual feature importances.

        """
        return self.accuracy(mask), self.importances(mask)

    def models(self, mask: np.ndarray) -> List[Mercs]:
        """Generate models for a given mask.
        
        Returns:
            A model train/test split.

        """
        key = mask.tobytes()
        # print([self._columns[i] for i, v in enumerate(mask) if v == 0])
        if key not in self._cache:
            self._cache[key] = list()
            code = self._code(mask)
            for (train, _) in self._folds:
                model = Mercs(**self._m_args)
                model.fit(train, nominal_attributes=self._nominal, m_codes=code)
                # compute SHAP values
                # model.avatar(train, keep_abs_shap=True, check_additivity=False)
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

        # shuffle rows and convert to mercs.
        df, _ = to_mercs(df.sample(frac=1))

        # get min_full rows from the dataframe
        full = df[df.notna().all(axis=1)].sample(min_full)
        rest = df[~df.index.isin(full.index)]

        # sample test and train
        test = rest.sample(self._test_size)
        train = rest[~rest.index.isin(test.index)].sample(
            self._n_samples - self._test_size - min_full
        )

        return pd.concat((full, train), axis=0), test


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

    def evaluate(self):
        """Evaluate on folds."""
        # generate a code
        code = self._code()
        # run the train, test splits.
        accuracies = list()
        for train, test in self._folds():
            test = np.nan_to_num(test)
            # learn model
            model = Mercs(**self._m_args)
            model.fit(train, nominal_attributes=self._nominal, m_codes=code)
            # make prediction for each model
            for m_code in model.m_codes:
                prediction = model.predict(test, q_code=m_code)
                truth = test[:, m_code == 1][:, 0]
                accuracies.append(accuracy_score(prediction, truth))
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
        for train, test in KFold(n_splits=self._n_folds).split(self._data):
            yield self._data[train], self._data[test]

    def _metadata(self):
        """Return metadata."""
        attributes = set(range(len(self._columns)))
        return dict(
            n_attributes=len(attributes),
            nominal_attributes=self._nominal,
            numeric_attributes=attributes - self._nominal,
        )


class ColumnSampler:
    """Column sampler."""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._df_notna = df.notna()

    @abstractmethod
    def sample(self) -> pd.DataFrame:
        pass

    @cached_property
    def to_sample(self) -> List[Label]:
        """Columns that need sampling."""
        return self._df_notna.columns[~self._df_notna.all()].to_list()

    @cached_property
    def not_to_sample(self) -> List[Label]:
        """Columns that don't need sampling."""
        return self._df_notna.columns[self._df_notna.all()].to_list()

    def is_valid(self, df: pd.DataFrame):
        """Check if set of columns is valid."""
        return df.all(axis=1).sum() > 10

    def make_result(self, sampled: List[Label]) -> pd.DataFrame:
        """Get result."""
        return pd.concat((self._df[self.not_to_sample], self._df[sampled]), axis=1)


class UniformColumnSampler(ColumnSampler):
    """Sample uniformly."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def sample(self) -> pd.DataFrame:
        """Sample a set of columns with a full row.
        
        Iteratively remove a column until valid solution found.
    
        """
        df = self._df_notna[self.to_sample]
        while len(df.columns) > 0:
            if self.is_valid(df):
                break
            rm = random.choice(df.columns)
            df = df.drop(rm, axis=1)
        return self.make_result(df.columns)


class WeightedColumnSampler(ColumnSampler):
    """Weigh samples by number of nans."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    def sample(self) -> pd.DataFrame:
        df = self._df_notna[self.to_sample]
        ws = self.weights[self.to_sample]
        while len(df.columns) > 0:
            if self.is_valid(df):
                break
            rm = random.choices(df.columns, weights=ws, k=1)
            df = df.drop(rm, axis=1)
            ws = ws.drop(rm)
        return self.make_result(df.columns)

    @cached_property
    def weights(self) -> pd.Series:
        weights = (~self._df_notna).sum(axis=0)
        weights[weights != 0] = np.log(weights[weights != 0])
        weights = normalize(weights)
        return weights


class SmartColumnSampler(WeightedColumnSampler):
    """Weighs by how often picked before."""

    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self._counts = pd.Series(0, index=df.columns)

    def sample(self) -> pd.DataFrame:
        df = self._df_notna[self.to_sample]
        ws = self.weights[self.to_sample]
        while len(df.columns) > 0:
            if self.is_valid(df):
                break
            rm = random.choices(df.columns, weights=ws, k=1)
            df = df.drop(rm, axis=1)
            ws = ws.drop(rm)
        return self.make_result(df.columns)

    @property
    def counts(self):
        """Normalized counts."""
        return normalize(self._counts)

    @property
    def smart_weights(self):
        """Combine weights with count."""
        return self.counts.add(super().weights) / 2
