import pandas as pd
import numpy as np
from pandas._typing import Label
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.base import is_classifier
from tqdm import tqdm
from sklearn.model_selection import (
    cross_val_score,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from .supervised import _shuffle_encode
from .settings import Settings


class RankingEvaluator:
    """Use ranking to select features and evaluate."""

    def __init__(
        self, estimator, max_features: int = 32, folds: int = 5, samples: int = 5000
    ) -> None:
        self.estimator = estimator
        self.max_features = max_features
        self.folds = folds
        self.samples = 5000
        self.scores = dict()

    def fit(self, data: pd.DataFrame, target: Label, ranking: Dict[Label, float]):
        ranked = sorted(ranking, key=lambda x: ranking[x], reverse=True)
        scores = dict()
        for i in range(self.folds):
            shuffled = _shuffle_encode(data)
            for n in tqdm(range(1, self.max_features), disable=not Settings.verbose):
                X = shuffled[ranked[:n]]
                s = np.mean(
                    cross_val_score(
                        self.estimator, X, shuffled[target], cv=self.splitter(data)
                    )
                )
                if i == 0:
                    scores[n] = {"scores": [s], "features": ranked[:n]}
                else:
                    scores[n]["scores"].append(s)
        self.scores = scores

    def splitter(self, data: pd.DataFrame):
        """Return a splitter with."""
        n = min(len(data), self.samples)
        t = int(n // 10)
        if is_classifier(self.estimator):
            splitter = StratifiedShuffleSplit
        else:
            splitter = ShuffleSplit
        return splitter(n_splits=5, test_size=t, train_size=n - t)

    # def select(self) -> Tuple[List[Label], float]:
    #     best = max(self.scores, key=lambda n: self.scores[n]["score"])
    #     return self.scores[best]


# class DatasetEvaluator:
#     """Dataset evaluator."""

#     def __init__(self, n_folds: int = 5, **configuration):
#         self._n_folds = n_folds
#         self._m_args = dict()
#         self._m_args.update(configuration)

#     def fit(self, df: pd.DataFrame, target: Optional[Label] = None):
#         # prepare data
#         data, nominal = to_mercs(df)
#         self._data = data.values
#         self._nominal = nominal
#         self._columns = df.columns
#         self._target = target
#         self._target_index = df.columns.get_loc(target)

#     def evaluate(self, get_importances=False):
#         """Evaluate on folds. """
#         # generate a code
#         code = self._code()
#         # run the train, test splits.
#         accuracies = list()
#         importances = list()
#         for train, test in self._folds():
#             test = np.nan_to_num(test)
#             # learn model
#             model = Mercs(**self._m_args)
#             model.fit(train, nominal_attributes=self._nominal, m_codes=code)
#             if get_importances:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore")
#                     model.avatar(train, keep_abs_shap=True, check_additivity=False)
#                     importances.append(np.sum(model.nrm_shaps, axis=0))
#             # make prediction for each model
#             for m_code in model.m_codes:
#                 prediction = model.predict(test, q_code=m_code)
#                 truth = test[:, m_code == 1][:, 0]
#                 # classification
#                 if self._target_index in self._nominal:
#                     accuracy = accuracy_score(truth, prediction)
#                 # or regression
#                 else:
#                     accuracy = mean_squared_error(truth, prediction)
#                 accuracies.append(accuracy)
#         if get_importances:
#             return np.mean(accuracies), np.mean(importances, axis=0)
#         return np.mean(accuracies)