from avatar.utilities import get_estimator
from inspect import Parameter
from scipy.sparse.construct import rand
import shap
import random
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pandas._typing import Label
from typing import Callable, Dict, List, Tuple, Type, Iterable, Union
from operator import itemgetter
from tqdm import tqdm
from trueskill import Rating, rate, rate_1vs1
from sklearn.preprocessing import normalize, LabelEncoder
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
Parameters = Union[str, Dict[str, "Parameter"]]


class Judge(ABC):
    """Base game judge."""

    @abstractmethod
    def evaluate(self, model, X, y) -> np.ndarray:
        pass

    @property
    def parameters(self) -> Parameters:
        return self.__class__.__name__

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


class Skill(ABC):
    """Base importance class.

    Should support retrieving a value and provide
    a method for aggregating a list of performances.

    """

    @property
    def value(self) -> float:
        pass

    @staticmethod
    def update(performances: List[Tuple["Skill", float]]) -> None:
        """Update skill with game performances.

        Args:
            performances: List of (Skill, performance) tuples.

        """
        pass

    def __str__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.value)


class AverageSkill(Skill):
    """Use average importances."""

    def __init__(self) -> None:
        self._values = list()

    @property
    def value(self) -> float:
        if len(self._values) == 0:
            return 0
        return np.mean(self._values)

    def add(self, value: float) -> None:
        self._values.append(value)


class TrueSkill(Skill):
    """Use trueskill to represent importances."""

    def __init__(self) -> None:
        self.rating = Rating()
        self.n = 0

    @property
    def value(self) -> float:
        """Compute value of TrueSkill importance."""
        return self.rating.mu - self.rating.sigma


class Pool(ABC):
    """Pool of players."""

    def __init__(self) -> None:
        """Initialise."""
        self.players = None

    def sample(self, size: int = 16, exploration: float = 0.0) -> List[Label]:
        """Sample a team.

        MinMax scale the skill before sampling.

        """
        # get players and weights
        p, w = map(list, zip(*((p, s.value) for p, s in self.players.items())))
        # scale the weights
        m = min(w)
        w = [v - m for v in w]
        # build team
        team = list()
        while len(team) < size:
            # generate index
            if random.random() < exploration:
                i = random.randint(0, len(w) - 1)
            else:
                i = random.choices(range(len(w)), weights=w, k=1)[0]
            # add to team
            team.append(p[i])
            # remove from lists
            del p[i]
            del w[i]
        return team

    @abstractmethod
    def initialise(self, players: Iterable[Label]) -> None:
        pass

    @abstractmethod
    def update(self, results: Dict[Label, float]):
        """Update ratings.

        Args:
            results: Mapping of players to performance in
                some game.

        """
        pass

    @property
    def skills(self) -> Dict[Label, float]:
        """Get the ranking of players."""
        return {player: skill.value for player, skill in self.players.items()}

    def __str__(self) -> str:
        return self.__class__.__name__[:-4]


class AveragePool(Pool):
    """Use average ranking."""

    def initialise(self, players: Iterable[Label]) -> None:
        self.players = {player: AverageSkill() for player in players}

    def update(self, results: Dict[Label, float]):
        for player, score in results.items():
            self.players[player].add(score)


class TruePool(Pool):
    """Use TrueSkill for aggregating ranks."""

    def __init__(self) -> None:
        super().__init__()

    def initialise(self, players: Iterable[Label]) -> None:
        self.players = {player: TrueSkill() for player in players}

    def update(self, results: Dict[Label, float]):
        # sort from high to low performance
        performances = sorted(results.items(), key=itemgetter(1), reverse=True)
        for i, (n1, s1) in enumerate(performances[:-1]):
            n2, s2 = performances[i + 1]
            # get players
            p1: TrueSkill = self.players[n1]
            p2: TrueSkill = self.players[n2]
            # get updated ratings
            r1, r2 = rate_1vs1(p1.rating, p2.rating, drawn=s1 == s2)
            # update skill objects
            p1.rating = r1
            p1.rating = r2

    @property
    def skills(self) -> Dict[Label, float]:
        """Get ranking.

        Min scale the ranking before returning.

        """
        ranking = super().skills
        lowest = min(ranking.values())
        return {player: value - lowest for player, value in ranking.items()}


class Game:
    """Represents a game that teams can play."""

    def __init__(
        self,
        estimator=None,
        judge: Judge = None,
        rounds: int = 1,
        samples: int = 1000,
    ) -> None:
        """

        Args:
            estimator: Model used.

        """
        self.estimator = estimator
        self.judge = judge or SHAPJudge()
        self.rounds = rounds
        self.samples = samples
        self.data = None
        self.target = None
        self.splitter = None

    def initialise(self, data: pd.DataFrame, target: Label) -> None:
        """Initialise game on some data."""
        self.data = data
        self.target = target
        # set the estimator
        if self.estimator is None:
            self.estimator = _default_estimator(data, target)
        # set the splitter
        if is_classifier(self.estimator):
            splitter = StratifiedShuffleSplit
        else:
            splitter = ShuffleSplit
        n = min(self.samples, len(data))
        t = n // 10
        self.splitter = splitter(n_splits=self.rounds, train_size=n - t, test_size=t)

    def play(self, team: List[Label]) -> Dict[Label, float]:
        """Let a team play the games and aggregate performance.

        Returns:
            A map of player names to their average contribution
            over the played games.

        """

        # get data for this team, shuffle and encode it. If the
        # encapsulating class already encoded, it will only
        # shuffle and not lose any speed. This call is then
        # mostly a safeguard.
        X = _shuffle_encode(self.data[team])
        y = self.data[self.target][X.index]

        # play our the rounds
        performances = list()
        for train, test in self.splitter.split(X, y):
            # get data
            Xtr, Xte = X.iloc[train], X.iloc[test]
            ytr, yte = y.iloc[train], y.iloc[test]
            # fit model
            self.estimator.fit(Xtr, ytr)
            # get performance and score on training data
            score = max(0, self.estimator.score(Xte, yte))
            perfs = self.judge.evaluate(self.estimator, Xte, yte)
            performances.append(perfs * (1 + score))
        performances = np.vstack(performances)

        # aggregate
        aggregated = np.maximum(performances.mean(axis=0) - performances.std(axis=0), 0)
        # aggregated = np.maximum(performances.mean(axis=0), 0)

        # map back to players
        return {player: aggregated[i] for i, player in enumerate(team)}

    @property
    def parameters(self) -> Parameters:
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


class Tournament:
    """Tournament for evaluating features."""

    def __init__(
        self,
        game: Game,
        pool: Pool,
        games: int,
        teamsize: Union[int, str],
        exploration: float,
    ) -> None:
        """Initialise tournament.

        Args:
            game: A game object.
            games: Number of games to play. (In the future, automated
                stopping criteria will be added.)
            teamsize: Number of players for each game. Can be a constant
                or a function that is applied on the number of total
                players in the pool.
            exploration: Exploration for selecting players.

        """
        self.game = game
        self.pool = pool
        self.games = games
        self.exploration = exploration
        self._teamsize = teamsize

        self.data = None
        self.target = None

    def initialise(self, data: pd.DataFrame, target: Label) -> None:
        """Initialise tournament with data and target."""
        self.data = data
        self.label = target
        # initialise pool
        self.pool.initialise(set(data.columns) - {target})
        # initialise the game
        self.game.initialise(data, target)

    def play(self):
        """Play the tournament."""
        for _ in tqdm(range(self.games), disable=not Settings.verbose):
            self.round()

    def round(self):
        """Play one round."""
        team = self.pool.sample(self.teamsize, self.exploration)
        performance = self.game.play(team)
        self.pool.update(performance)

    @property
    def teamsize(self) -> int:
        if self._teamsize == "sqrt":
            return int(np.sqrt(len(self.data.columns)))
        else:
            return self._teamsize

    @property
    def results(self) -> Dict[Label, float]:
        score = self.pool.skills
        total = sum(score.values())
        if total > 0:
            return {player: skill / total for player, skill in score.items()}
        return dict()

    @property
    def parameters(self) -> Parameters:
        """Get parameters of this model."""
        return {
            "type": self.__class__.__name__,
            "game": self.game.parameters,
            "pool": str(self.pool),
            "games": self.games,
            "team": self._teamsize,
            "exploration": self.exploration,
        }

    def __str__(self):
        return "{}(g={}, p={}, n={}, t={}, e={})".format(
            self.__class__.__name__,
            self.game,
            self.pool,
            self.games,
            self._teamsize,
            int(self.exploration * 100),
        )


class AnnealingTournament(Tournament):
    """Cool down the team size.

    Start with all features in one team and decrease
    after playing more itertations.

    """

    def initialise(self, data: pd.DataFrame, target: Label):
        super().initialise(data, target)
        self.current_round = 0
        self.teamsizes = list(
            map(int, np.geomspace(super().teamsize, len(data.columns) - 1, self.games))
        )[::-1]

    def play(self):
        """Play tournament."""
        pbar = tqdm(total=len(self.teamsizes), disable=not Settings.verbose)
        while self.teamsize > 0:
            pbar.update()
            pbar.set_description(str(self.teamsize))
            self.round()
            self.current_round += 1

    @property
    def teamsize(self) -> int:
        """Get next team size."""
        if self.current_round < len(self.teamsizes):
            return self.teamsizes[self.current_round]
        return 0


class LegacyTournament(Tournament):
    """Tournament with old AVATAR settings.

    – only exploration
    – random team size in each iteration (~ uniform sampling)
    – no stratification
    – SHAP judging
    – average skill
    – one round

    """

    def __init__(self, games: int) -> None:
        # initialise tournament
        super().__init__(
            game=None,
            pool=AveragePool(),
            games=games,
            teamsize=lambda x: random.randint(1, x),
            exploration=1.0,
        )

    def initialise(self, data: pd.DataFrame, target: Label) -> None:
        self.data = data
        self.target = target
        # initialise pool
        self.pool.initialise(set(data.columns) - {target})
        # initialise the game
        n = min(1000, len(data))
        t = int(n // (1 / 0.2))
        self.game = Game(
            _default_estimator(data, target), SHAPJudge(), rounds=1, samples=1000
        )
        self.game.initialise(data, target)

    @property
    def teamsize(self) -> int:
        """Sample random team size each iteration."""
        return self._teamsize(len(self.data.columns) - 1)

    @property
    def parameters(self) -> Parameters:
        """Get parameters of this model."""
        return {
            "type": self.__class__.__name__,
            "game": self.game.parameters,
            "pool": str(self.pool),
            "games": self.games,
            "team": "random",
            "exploration": self.exploration,
        }

    def __str__(self):
        return "{}(g={}, p={}, n={}, t={}, e={})".format(
            self.__class__.__name__,
            self.game,
            self.pool,
            self.games,
            "random",
            0,
        )


def _default_estimator(data: pd.DataFrame, target: Label):
    """Get default estimator."""
    if data[target].dtype.name in ["object", "category"]:
        return DecisionTreeClassifier(max_depth=4)
    else:
        return DecisionTreeRegressor(max_depth=4)


def _shuffle_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode dataframe.

    Returns:
        A new dataframe with columns label encoded and an
        empty dictionary.

    """
    df = df.sample(frac=1)
    for c in df:
        if df[c].dtype.name in ["object", "category"]:
            df[c] = LabelEncoder().fit_transform(df[c])
    return df.fillna(-1)
