import random
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Iterable, Union, Hashable
from operator import itemgetter
from tqdm.auto import tqdm
from trueskill import Rating, rate_1vs1
from sklearn.model_selection import (
    BaseCrossValidator,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from .settings import Settings
from avatar.evaluate import Game, Result


Splitter = Union[BaseCrossValidator, ShuffleSplit, StratifiedShuffleSplit]
Parameters = Union[str, Dict[str, "Parameters"]]


class Pool(ABC):
    """Pool of players."""

    def __init__(self, burn: bool = False) -> None:
        """Initialise.

        Args:
            players: A list of player names.

        """
        self.players: List[Hashable] = list()
        self.burn = burn

    def initialise(self, players: Iterable[Hashable]) -> None:
        """Initialise the pool with players."""
        self.players = list(players)

    def sample(self, size: int = 16, exploration: float = 1.0) -> List[Hashable]:
        """Sample a team.

        Sample either randomly (explore) or weighed
        by rating (exploit) for every player.

        Args:
            size: Number of players to sample.
            exploration: Exploration rate.

        Returns:
            A list of players.

        """
        # whole team needs to be selected
        if size >= len(self.players):
            return self.players[:]
        # get players and weights
        p, w = map(list, zip(*((p, self.rating(p) + 0.05) for p in self.players)))
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
    def update(self, results: Result):
        """Update ratings.

        Args:
            results: Mapping of players to performance in
                some game.

        """
        pass

    @abstractmethod
    def rating(self, player: Hashable) -> float:
        """Get the rating of one player."""
        pass

    def ratings(self) -> Dict[Hashable, float]:
        """Get the ratings of all players."""
        return {player: self.rating(player) for player in self.players}

    def __str__(self) -> str:
        return self.__class__.__name__[:-4]


class AveragePool(Pool):
    """Use average ranking."""

    def __init__(self) -> None:
        super().__init__()
        self._scores = dict()
        self._counts = dict()

    def initialise(self, players: Iterable[Hashable]) -> None:
        super().initialise(players)
        self._scores = {player: list() for player in players}
        self._counts = {player: 0 for player in players}

    def update(self, result: Result):
        """Update mean score of each player."""
        for player, score in result.performances.items():
            self._counts[player] += 1
            self._scores[player].append(score)

    def rating(self, player: Hashable) -> float:
        """Compute rating as average."""
        if self._counts[player] == 0:
            return 1
        return sum(self._scores[player]) / self._counts[player]


class TruePool(Pool):
    """Use TrueSkill for aggregating ranks."""

    def initialise(self, players: Iterable[Hashable]) -> None:
        super().initialise(players)
        self.players = {player: Rating() for player in players}

    def update(self, result: Result):
        # sort from high to low performance
        performances = sorted(result.performances.items(), key=itemgetter(1))[::-1]
        for i, (n1, s1) in enumerate(performances[:-1]):
            n2, s2 = performances[i + 1]
            # get players
            p1: Rating = self.players[n1]
            p2: Rating = self.players[n2]
            # get updated ratings
            r1, r2 = rate_1vs1(p1, p2, drawn=(s1 == s2))
            # update skill objects
            self.players[n1] = r1
            self.players[n2] = r2

    def rating(self, player: Hashable) -> float:
        """Compute rating."""
        return self.players[player].mu - self.players[player].sigma


class Tournament:
    """Tournament for evaluating features."""

    def __init__(
        self,
        game: Game,
        pool: Pool,
        games: int,
        size: Union[int, Callable[[int], int]],
        exploration: float,
    ) -> None:
        """Initialise tournament.

        Args:
            game: A game object.
            games: Number of games to play.
            size: Number of players for each game. Can be a constant
                or a function that is applied on the number of total
                players in the pool.
            exploration: Exploration for selecting players.

        """
        self.game = game
        self.pool = pool
        self.games = games
        self.exploration = exploration
        self.size = size
        self.data = None
        self.target = None
        self.current = 0
        self.results: list[Result] = list()

    def initialise(self, data: pd.DataFrame, target: Hashable) -> None:
        """Initialise tournament with data and target."""
        self.data = data
        self.label = target
        self.pool.initialise(set(data.columns) - {target})
        self.game.initialise(data, target)
        self.current = 0

    def play(self):
        """Play the tournament."""
        for _ in tqdm(range(self.games), disable=not Settings.verbose, leave=False):
            self.round()
            self.current += 1

    def round(self):
        """Play one round."""
        team = self.pool.sample(self.teamsize, self.exploration)
        perf = self.game.play(team)
        self.results.append(perf)
        self.pool.update(perf)

    @property
    def teamsize(self) -> int:
        if isinstance(self.size, int):
            return self.size
        elif callable(self.size):
            return self.size(len(self.data.columns))

    @property
    def ratings(self) -> Dict[Hashable, float]:
        score = self.pool.ratings()
        total = sum(score.values())
        if total > 0:
            return {player: skill / total for player, skill in score.items()}
        return dict()

    @property
    def parameters(self) -> Parameters:
        """Get parameters of this model."""
        return {
            "game": self.game.parameters,
            "pool": str(self.pool),
            "games": self.games,
            "team": self.size,
            "exploration": self.exploration,
        }

    def __str__(self):
        return "{}(g={}, p={}, n={}, t={}, e={})".format(
            self.__class__.__name__,
            self.game,
            self.pool,
            self.games,
            self.size,
            int(self.exploration * 100),
        )


# class AnnealingTournament(Tournament):
#     """Cool down the team size.

#     Start with all features in one team and decrease
#     after playing more iterations.

#     """

#     def initialise(self, data: pd.DataFrame, target: Hashable):
#         super().initialise(data, target)
#         self.current_round = 0
#         self.teamsizes = list(
#             map(int, np.geomspace(super().teamsize, len(data.columns) - 1, self.games))
#         )[::-1]

#     def play(self):
#         """Play tournament."""
#         pbar = tqdm(total=len(self.teamsizes), disable=not Settings.verbose)
#         while self.teamsize > 0:
#             pbar.update()
#             pbar.set_description(str(self.teamsize))
#             self.round()
#             self.current_round += 1

#     @property
#     def teamsize(self) -> int:
#         """Get next team size."""
#         if self.current_round < len(self.teamsizes):
#             return self.teamsizes[self.current_round]
#         return 0


# class LegacyTournament(Tournament):
#     """Tournament with old AVATAR settings.

#     – only exploration
#     – random team size in each iteration (~ uniform sampling)
#     – no stratification
#     – SHAP judging
#     – average skill
#     – one round

#     """

#     def __init__(self, games: int) -> None:
#         # initialise tournament
#         super().__init__(
#             game=None,
#             pool=AveragePool(),
#             games=games,
#             teamsize=lambda x: random.randint(1, x),
#             exploration=1.0,
#         )

#     def initialise(self, data: pd.DataFrame, target: Hashable) -> None:
#         self.data = data
#         self.target = target
#         # initialise pool
#         self.pool.initialise(set(data.columns) - {target})
#         # initialise the game
#         n = min(1000, len(data))
#         t = int(n // (1 / 0.2))
#         self.game = Game(
#             _default_estimator(data, target), SHAPJudge(), rounds=1, samples=1000
#         )
#         self.game.initialise(data, target)

#     @property
#     def teamsize(self) -> int:
#         """Sample random team size each iteration."""
#         return self._teamsize(len(self.data.columns) - 1)

#     @property
#     def parameters(self) -> Parameters:
#         """Get parameters of this model."""
#         return {
#             "type": self.__class__.__name__,
#             "game": self.game.parameters,
#             "pool": str(self.pool),
#             "games": self.games,
#             "team": "random",
#             "exploration": self.exploration,
#         }

#     def __str__(self):
#         return "{}(g={}, p={}, n={}, t={}, e={})".format(
#             self.__class__.__name__,
#             self.game,
#             self.pool,
#             self.games,
#             "random",
#             0,
#         )


# def _default_estimator(data: pd.DataFrame, target: Hashable):
#     """Get default estimator."""
#     if data[target].dtype.name in ["object", "string", "category"]:
#         return DecisionTreeClassifier(max_depth=4)
#     else:
#         return DecisionTreeRegressor(max_depth=4)


# def _encode(df: pd.DataFrame) -> pd.DataFrame:
#     """Ensure that all columns are categorical."""
#     pass


# def _shuffle_encode(df: pd.DataFrame) -> pd.DataFrame:
#     """Encode dataframe.

#     Returns:
#         A new dataframe with columns label encoded and an
#         empty dictionary.

#     """
#     df = df.sample(frac=1)
#     for c in df:
#         if df[c].dtype.name in ["object", "category"]:
#             df[c] = HashableEncoder().fit_transform(df[c])
#     return df.fillna(-1)
