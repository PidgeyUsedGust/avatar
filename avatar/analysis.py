"""Analyse quality of a dataset by machine learning.

We use MERCS as a backend as this allows us to have targeted
and targetless wrangling in one clean framework.

"""
import random
import pandas as pd
import numpy as np
from pandas._typing import Label
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from functools import cached_property
from collections import defaultdict
from mercs.core import Mercs
from .utilities import normalize


_merc_config = dict(
    # Induction
    max_depth=1,
    selection_algorithm="default",
    nb_targets=1,
    nb_iterations=1,
    n_jobs=1,
    # Inference
    inference_algorithm="own",
    prediction_algorithm="mi",
    max_steps=8,
)
"""Configuration for MERCS.

Mainly added for experimental purposes. Only change this if you
know what you're doing."""


class Analyzer:
    """Base analyser."""

    @abstractmethod
    def predictive_accuracy(self) -> float:
        pass

    @abstractmethod
    def feature_importances(self) -> Dict[str, float]:
        pass


class SupervisedAnalyzer(Analyzer):
    """Analysis with respect to a target column."""

    def __init__(self, df: pd.DataFrame, target: Label):
        self._df = df
        self._target = target
        self._model = None

    def analyze(self, iterations: int = 1):
        # prepare data
        data, nominal = to_mercs(self._df)
        custom_m_code = to_m_codes(self._df, self._target)
        # train model
        self._model = Mercs(**_merc_config)
        self._model.fit(data, nominal_attributes=nominal, m_codes=custom_m_code)

    def feature_importances(self) -> Dict[str, float]:
        """Get feature importances.
        
        Returns:
            A mapping from column names to feature importances.

        """
        accuracies = np.sum(self._model.m_fimps, axis=0)
        importance = {
            column: accuracies[i] for i, column in enumerate(self._df.columns)
        }
        return importance

    def predictive_accuracy(self):
        """Predictive accuracy."""
        print(self._model.m_score)


class UnsupervisedAnalyzer:
    """Analyse a dataset."""

    def __init__(self, df: pd.DataFrame):
        """
        
        Args:
            target: Target column for prediction.

        """

        self._df = df
        self._model = None

        # train the model
        self.analyze()

    def analyze(self):
        data, nominal = to_mercs(self._df)
        self._model = Mercs(**_merc_config)
        self._model.fit(data, nominal_attributes=nominal)

    def feature_importances(self) -> Dict[str, float]:
        """Get feature importances.
        
        Returns:
            A mapping from column names to feature importances.

        """
        accuracies = np.sum(self._model.m_fimps, axis=0)
        importance = {
            column: accuracies[i] for i, column in enumerate(self._df.columns)
        }
        return importance

    def predictive_accuracy(self):
        """Predictive accuracy."""
        print(self._model.m_score)


class FeatureSelector(ABC):
    """Base feature selector."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

        self.select()

    @abstractmethod
    def select(self) -> List[Label]:
        pass

    @property
    @abstractmethod
    def selected(self) -> List[Label]:
        pass

    @property
    def not_selected(self) -> List[Label]:
        pass


# class SimilarFeatureSelector(FeatureSelector):
#     """Select features based on making the same predictions."""

#     def select(self) -> List[Label]:
#         pass

#     def select_correlation(self, df: pd.)


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
        # print(df.all(axis=1).sum())
        # return df.all(axis=1).any()
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


def to_mercs(df: pd.DataFrame) -> Tuple[np.ndarray,]:
    """Encode dataframe for MERCS.
    
    We assume numerical values will have a numerical dtype and convert
    everything else to nominal integers.

    """
    new = pd.DataFrame()
    nom = set()
    for i, column in enumerate(df):
        if df[column].dtype.name in ["category", "object"]:
            new[column] = df[column].astype("category").cat.codes.replace(-1, np.nan)
            nom.add(i)
        else:
            new[column] = df[column]
    return new.values, nom


def to_m_codes(df: pd.DataFrame, target: Label):
    """Generate m_codes for a target."""
    m_codes = np.zeros((1, len(df.columns)))
    m_codes[0, df.columns.get_loc(target)] = 1
    return m_codes


def available_models() -> List[str]:
    from importlib import import_module

    # list of possible classes
    possible = [
        ("xgboost", "XGBClassifier", "XGB"),
        ("lightgbm", "LGBMClassifier", "LGBM"),
        ("catboost", "CatBoostClassifier", "CB"),
        ("wekalearn", "RandomForestClassifier", "weka"),
    ]
    models = list()
    for package, classname, short in possible:
        try:
            module = import_module(package)
            class_ = getattr(module, classname)
        except:
            pass
        finally:
            models.append(short)
    return models

