import shap
import random
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, normalize
from trueskill import Rating, rate, quality
from pandas._typing import Label
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Tuple, Dict, Protocol
from collections import defaultdict
from .expand import Tracker
from .utilities import encode_label, encode_onehot


class Task:
    """A prediction task."""

    def __init__(self, X: List[Label], y: Label):
        self.X = X
        self.y = y

    def slice(self, df: pd.DataFrame, n: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Slice out data.

        Args:
            n: If > 0, sample this many rows.

        Returns:
            A tuple of X and y data without
            missing values in y.

        """
        df = df.dropna(subset=[self.y])
        if n > 0:
            df = df.sample(min(n, len(df.index)))
        return df[self.X], df[self.y]


class TaskGenerator(ABC):
    """Abstract task generator."""

    @abstractmethod
    def generate(self, df: pd.DataFrame, tracker: Tracker) -> List[Task]:
        pass


class RoundRobinGenerator(TaskGenerator):
    """Generate tasks round robin."""

    def generate(self, df: pd.DataFrame, tracker: Tracker) -> List[Task]:
        tasks = list()
        columns = set(df.columns)
        todo = list(df.columns)
        while len(todo) > 0:
            column = todo[0]
            family = set(tracker.family(column)) & columns
            for y in family:
                todo.remove(y)
                # # don't consider columns with missing values.
                # if df[y].isna().any():
                #     continue
                # initialise X with everyone not in the family
                X = columns - family
                # check who from the family we can add
                for other in family - {y}:
                    if not share_characters(y, other, tracker):
                        X.add(other)
                # have full task
                tasks.append(Task(X, y))
            # break
        return tasks


class RandomGenerator(TaskGenerator):
    """Generate tasks randomly."""

    def __init__(self, n: int = 0, p: float = 0.5):
        """

        Args:
            n: Number of random tasks to generate. If
                set to 0, generate random task for
                every column. If larger than the number
                of columns,
            p: Probability that each column is included.

        """
        self.n = n
        self.p = p

    def generate(self, df: pd.DataFrame, tracker: Tracker) -> List[Task]:
        # first generate targets
        if self.n == 0:
            targets = list(df.columns)
        elif self.n < len(df.columns):
            targets = random.sample(df.columns, self.n)
        else:
            n, r = divmod(self.n, len(df.columns))
            targets = n * list(df.columns) + random.sample(df.columns, r)
        # generate tasks
        tasks = list()
        for y in targets:
            mask = np.random.rand(len(df.columns))
            mask = mask < self.p
            family = tracker.family(y)
            X = list()
            for c in df.columns[mask]:
                if c not in family or not share_characters(c, y, tracker):
                    X.append(c)
            tasks.append(Task(X, y))
        return tasks


class TaskEvaluator:
    """Evaluate a single task."""

    def __init__(self, models, importance: str, sample: int = 0, onehot=None):
        """

        Args:
            modesl: Model following sklearn fit-predict interface
                to use for training.
            importance: Method for extracting importances from
                an sklearn model. Either "shap" or "mdi".
            sample: If > 0, sample this many rows for each task.
            onehot: If given, will onehot encode the categorical
                columns and use this function to aggregate the
                results.

        """
        self.models = models
        self.importance = importance
        self.sample = sample
        self.onehot = onehot

    def evaluate(self, task: Task, data: pd.DataFrame) -> Tuple[List[float], List[str]]:
        """Evaluate a task.

        Returns:
            A tuple of (I, F) where F[i] is a feature with
            importance I[i].

        """
        # create sliced data and encode
        X, y = task.slice(data, n=self.sample)
        if self.onehot is None:
            Xe, M = encode_label(X)
        else:
            Xe, M = encode_onehot(X)
        # get importances of encoded data
        Ie = self.extract(Xe, y)
        importances = dict()
        for column in X.columns:
            # importance directly available
            if column in Ie:
                importances[column] = Ie[column]
            # need to aggregate and have some columns
            elif len(M[column]) > 0:
                importances[column] = self.onehot(Ie[c] for c in M[column])
            # was removed in pruning the 1H
            else:
                importances[column] = 0
        return importances

    def extract(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        # fit model
        if y.dtype.name in ["object", "bool", "category"]:
            model = self.models[0]
        else:
            model = self.models[1]
        model.fit(X, y)
        # get vector of importances
        if self.importance == "shap":
            raw = extract_treeshap(model, X)
        else:
            raw = extract_default(model, X)
        # map column names to scores
        return {X.columns[i]: score for i, score in enumerate(raw)}


class TaskAggregator(ABC):
    """Aggregate importances."""

    @abstractmethod
    def aggregate(self, importances: List[Dict[str, float]]):
        pass


class TrueSkillAggregator(TaskAggregator):
    """Use TrueSkill for extraction.

    The following modes are available:

     * All pairwise battles between features.
     * Only pairwise battles between neighbouring
       features.

    """

    def __init__(self, battles: str = "neighbours", warm_start: bool = False):
        self.mode = battles
        self.warm = warm_start

    def aggregate(self, data: List[Dict[str, float]]):
        """Aggregate individual importances.

        Args:
            data: List of the I, F pairs for individual tasks.

        Returns:


        """

        # group
        importances = defaultdict(list)
        for scores in data:
            for name, score in scores.items():
                importances[name].append(score)

        # initialise features
        if not self.warm:
            ratings = {feature: Rating() for feature in importances}
        else:
            ratings = dict()
            for feature, scores in importances.items():
                ratings[feature] = Rating(np.mean(scores), np.std(scores))

        # go over the rankings and generate games
        for scores in data:
            # generate battles
            if self.mode == "all":
                battles = self.battles_1v1_all(scores)
            elif self.mode == "neighbours":
                battles = self.battles_1v1_neighbours(scores)
            # run them
            for battle, rank in battles:
                # get ratings of players in battle
                battle_rating = [{p: ratings[p] for p in team} for team in battle]
                # execute battle
                for team in rate(battle_rating, rank):
                    for player in team:
                        ratings[player] = team[player]

        # compute score
        return {f: r.mu - r.sigma for f, r in ratings.items()}

    def battles_1v1_all(self, scores: Dict[str, float]):
        battles = list()
        for p1, p2 in combinations(scores, 2):
            r1 = scores[p1]
            r2 = scores[p2]
            if r1 > r2:
                rank = [0, 1]
            elif r1 == r2:
                rank = [0, 0]
            else:
                rank = [1, 0]
            battles.append(([(player1,), (player2,)], rank))
        return battles

    def battles_1v1_neighbours(self, scores: Dict[str, float]):
        ranking = sorted(scores.keys(), key=scores.get, reverse=True)
        battles = list()
        for i, p1 in enumerate(ranking[:-1]):
            p2 = ranking[i + 1]
            s1 = scores[p1]
            s2 = scores[p2]
            if s1 > s2:
                rank = [0, 1]
            elif s1 == s2:
                rank = [0, 0]
            else:
                rank = [1, 0]
            battles.append(([(p1,), (p2,)], rank))
        return battles

    # def battles_ffa(self, ranking):
    #     return [[(player,) for player in ranking[:200]]]


def extract_default(model, X) -> np.ndarray:
    """Get importances from model."""
    return model.feature_importances_


def extract_treeshap(model, X) -> np.ndarray:
    """Get TreeSHAP importances."""
    values = np.abs(np.array(shap.TreeExplainer(model).shap_values(X)))
    # if nominal target, sum across target
    if len(values.shape) == 3:
        values = np.sum(values, axis=0)
    # normalise to [0, 1]
    avg_shap = np.mean(values, axis=0)
    nrm_shap = np.squeeze(normalize(avg_shap.reshape(1, -1), norm="l1"))
    return nrm_shap


def share_characters(x: str, y: str, tracker: Tracker) -> bool:
    """Check if x and y possibly share characters.

    Check if it is possible that these columns
    share characters by looking for their LCA
    in the tracker and how they were generated
    from there.

    """
    transformations = tracker.lca_transformations(x, y)
    # if None, definitely share information
    if transformations is None:
        return True
    # else we can unpack
    (name1, arg1), (name2, arg2) = transformations
    # can't guarantee for different transformations
    if name1 != name2:
        return True
    # can't guarantee for different arguments of same
    # transformation, except in some cases
    if arg1 != arg2:
        if name1 in ["ExtractBoolean"]:
            return False
        return True
    # same transformation with same arguments
    # always disjoint
    return False