"""Analyse quality of a dataset by machine learning.

We use MERCS as a backend as this allows us to have targeted
and targetless wrangling in one clean framework.

"""
import pandas as pd
import numpy as np
from pandas._typing import Label
from typing import Dict
from mercs.core import Mercs
from .utilities import to_mercs


_merc_config = dict(
    # Induction
    max_depth=4,
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
    """Analyse a dataset."""

    def __init__(self, df: pd.DataFrame, target: Label = None):
        """
        
        Args:
            target: Target column for prediction.

        """

        self._df = df
        self._target = target
        self._model = None

        # get target index
        if target:
            self._m_codes = (df.columns == target).astype(int).reshape(1, -1)
        else:
            self._m_codes = None

        # train the model
        self._train()

    def _train(self):
        # convert
        data, nominal = to_mercs(self._df)
        # train model
        self._model = Mercs(**_merc_config)
        self._model.fit(data, nominal_attributes=nominal, m_codes=self._m_codes)

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
