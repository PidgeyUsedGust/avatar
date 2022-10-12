"""Evaluating of a set of features."""
import shap
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Sequence, Union, Hashable
from sklearn.preprocessing import normalize, robust_scale
from sklearn.model_selection import (
    BaseCrossValidator,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import is_classifier
from sklearn.inspection import permutation_importance
from .settings import Settings
from .utilities import estimator_to_string


Splitter = Union[BaseCrossValidator, ShuffleSplit, StratifiedShuffleSplit]
Parameter = Union[str, Union[str, "Parameter"]]


class Judge(ABC):
    """Base game judge."""

    @abstractmethod
    def evaluate(self, model, X, y) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__[:-5]


class DefaultJudge(Judge):
    """Use the model default."""

    def evaluate(self, model, X, y) -> np.ndarray:
        return model.feature_importances_


class PermutationJudge(Judge):
    """Use permutation feature importance."""

    def __init__(self, n_repeats: int = 5) -> None:
        self.n_repeats = n_repeats

    def evaluate(self, model, X, y) -> np.ndarray:
        return permutation_importance(
            model, X, y, n_repeats=self.n_repeats, n_jobs=Settings.n_jobs
        ).importances_mean


class SHAPJudge(Judge):
    """Use SHAP values."""

    def evaluate(self, model, X, y) -> np.ndarray:
        values = np.abs(np.array(shap.TreeExplainer(model).shap_values(X)))
        # if nominal target, sum across target
        if len(values.shape) == 3:
            values = np.sum(values, axis=0)
        # normalise to [0, 1]
        avg_shap = np.mean(values, axis=0)
        nrm_shap = np.squeeze(normalize(avg_shap.reshape(1, -1), norm="l1"))
        return nrm_shap


class Result:
    """Game result."""

    def __init__(self, team: Sequence[Hashable]) -> None:
        self._team = team
        self._performances = list()
        self._scores = list()

    def update(self, performance: np.ndarray, score: float) -> None:
        self._performances.append(performance)
        self._scores.append(score)

    @property
    def performances(self) -> Dict[Hashable, float]:
        """Get performances of players."""
        # make into numpy
        performances = np.vstack(self._performances)
        scores = np.array(self._scores)
        # scale performances by scores with broadcasting
        scaled = performances * (1 + scores)[:, None]
        # compute average
        avg = np.mean(scaled, axis=0)
        std = np.std(scaled, axis=0)
        # return scaled performance
        return {player: max(0, avg[i] - std[i]) for i, player in enumerate(self._team)}

    @property
    def score(self) -> float:
        """Get average score."""
        return np.mean(self._scores)

    @property
    def json(self) -> Dict[str, Any]:
        return {"score": self.score, "performances": self.performances}


class Game:
    """Represents a game that teams can play."""

    def __init__(
        self, estimator=None, judge: Judge = None, rounds: int = 1, samples: int = 1000
    ) -> None:
        """

        Args:
            estimator: Model used.

        """
        self.estimator = estimator
        self.judge = judge or DefaultJudge()
        self.rounds = rounds
        self.samples = samples
        self.data = None
        self.target = None
        self.splitter = None
        self.played: int = 0

    def initialise(self, data: pd.DataFrame, target: Hashable) -> None:
        """Initialise game on some data.

        Args:
            data: A dataframe that does not contain string
                or object columns.
            target: Column to be predicted.

        """

        # make sure that data is encoded
        self.data = encode(data, target)
        self.target = target

        # set the estimator
        if self.estimator is None:
            self.estimator = default_estimator(data, target)

        # set the splitter
        if is_classifier(self.estimator):
            splitter = StratifiedShuffleSplit
        else:
            splitter = ShuffleSplit
        n = min(self.samples, len(data))
        t = n // 10
        self.splitter = splitter(n_splits=self.rounds, train_size=n - t, test_size=t)

        # reset number of times played
        self.played = 0

    def play(self, team: Iterable[Hashable]) -> Result:
        """Let a team play the games and aggregate performance.

        Returns:
            A `Result` object.

        """

        # initialise result
        result = Result(team)

        # get data for this team
        X = self.data[list(team)]
        y = self.data[self.target][X.index]

        # play our the rounds
        for train, test in self.splitter.split(X, y):
            # get data
            Xtr, Xte = X.iloc[train], X.iloc[test]
            ytr, yte = y.iloc[train], y.iloc[test]
            # fit model
            self.estimator.fit(Xtr, ytr)
            # get performance and score on training data
            score = max(0, self.estimator.score(Xte, yte))
            perfs = self.judge.evaluate(self.estimator, Xte, yte)
            # update result
            result.update(perfs, score)

        # increase count
        self.played += 1

        # map back to players
        return result

    @property
    def parameters(self) -> Parameter:
        return {
            "estimator": estimator_to_string(self.estimator),
            "judge": self.judge.__class__.__name__,
            "rounds": self.rounds,
            "samples": self.samples,
        }

    def __str__(self) -> str:
        return "Game(e={}, j={}, r={}, s={})".format(
            estimator_to_string(self.estimator), self.judge, self.rounds, self.samples
        )


def default_estimator(data: pd.DataFrame, target: Hashable, depth: int = 4):
    """Get default estimator."""
    if data[target].dtype.name in ["object", "string", "category"]:
        return DecisionTreeClassifier(max_depth=depth)
    else:
        return DecisionTreeRegressor(max_depth=depth)


def encode(df: pd.DataFrame, target: Hashable) -> pd.DataFrame:
    """Encode the dataframe for learning.

    Returns:
        A dataframe containing only numerical values.

    """
    df = df.copy()
    for column in df:
        if column == target and df[column].dtype == "float64":
            df[column] = np.sign(df[column]) * np.log(np.abs(df[column]) + 1)
        elif df[column].dtype in ["category", "object", "string"]:
            df[column] = df[column].factorize()[0]
        elif df[column].dtype == "bool":
            df[column] = df[column].astype(int)
        elif df[column].dtype in ["float64"]:
            df[column] = np.sign(df[column]) * np.log(np.abs(df[column]) + 1)
    df = df.fillna(0)
    return df
