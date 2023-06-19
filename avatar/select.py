"""Feature selection.

Two types of selection are performed.

 * Filters will use intrinsic properties of a single column
   in order to quickly remove columns for consideration.

"""
from functools import cached_property
from avatar.ranking import Tournament
from avatar.utilities import divide, split
import random
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Optional,
    Tuple,
    List,
    Callable,
    Hashable,
    Set,
)
from math import ceil
from tqdm import tqdm
from .evaluate import Game, SHAPJudge, default_estimator


np.random.seed(1337)
random.seed(1337)


class Selector:
    """Base feature selector."""

    def __init__(self, evaluator: Game = None, games: int = 1000):
        """

        Args:
            evaluator: Feature evaluator used to perform games.
            games: Number of games the selector is allowed to
                play in total.

        """
        self._evaluator = evaluator or Game(rounds=4)
        self._games = games
        self._results = dict()

    def fit(
        self,
        df: pd.DataFrame,
        target: Optional[Hashable] = None,
        start: Optional[List[Hashable]] = None,
    ):
        """Fit selector.

        Args:
            df: Dataframe, ideally without string and object features.
                All non-numerical features are label encoded and data
                is not shuffled, so this might result in very biased
                results.
            target: Column to be predicted. Can be categorical.
            start: Initial list of features to include. By default, don't
                include any features.

        """
        self._df = df
        self._target = target
        self._start = start or list()
        self._evaluator.initialise(df, target)
        self.run()

    @abstractmethod
    def run(self):
        """Rank features.

        If this method sets the `self._results` dictionary,
        a select method does not have to implemented.

        """
        pass

    @property
    def budget(self) -> int:
        """Number of games left to play."""
        return self._games - self._evaluator.played

    def select(self) -> List[Hashable]:
        if len(self._results) > 0:
            return list(max(self._results, key=self._results.get))


class TournamentSelector(Selector):
    """Use tournament."""

    def __init__(self, tournament: Tournament, features: int = 32):
        """

        Args:
            tournament: Tournament used to rank games.
            features: Range of number of features to explore in ranking.

        """
        super().__init__(tournament.game, tournament.games)
        self.tournament = tournament
        self.features = features

    def run(self):
        """Perform ranking."""
        # perform ranking
        self.tournament.initialise(self._df, self._target)
        self.tournament.play()
        # initialise final evaluator
        self._evaluator.initialise(self._df, self._target)
        self._evaluator.estimator = default_estimator(self._df, self._target, 8)
        # get best features
        top = sorted(
            self.tournament.results, key=self.tournament.results.get, reverse=True
        )
        scores = [self._evaluator.play(top[:n]) for n in range(1, 1 + self._games)]
        for top, score in zip(top, scores):
            print(top, score.score)

    @property
    def required_depth(self) -> int:
        return ceil(np.log2(self._games))


class ChunkedSFFSelector(Selector):
    """Chunked Sequential Forward Floating Selection.

    The forward pass of SFFS becomes very slow for
    a large number of features.

    In chunked SFFS, we first divide the remaining
    features in chunks of a certain size and learn
    a model with `current | chunk` for each chunk.

    We then add the best chunk to the features
    and perform a backwards step.

    """

    def __init__(
        self,
        evaluator: Game = None,
        games: int = 1000,
        chunks: int = 16,
        add: str = "chunk",
    ):
        """

        Args:
            chunk: Number of chunks.
            add: {_chunk_, _best_} Method for adding chunks to the
                current set. If _chunk_, add the best chunk. If
                _best_, add the best feature from each chunk.

        """
        super().__init__(evaluator, games)
        self._chunks = chunks
        self._add = add

    def run(self):
        """Perform SFFS."""

        # get list of current features and remainder
        current = self._start
        remainder = self.remainder(current)

        # keep track of best team encountered, which is returned
        results = dict()

        while self.budget > 0:

            # shuffle the remaining features
            random.shuffle(remainder)

            # decide on size of each chunk
            size = max(1, ceil(len(remainder) / self._chunks))

            # if the budget is smaller than the minimal number of
            # games to be played, we swich to foward selection
            if self.budget < (self._chunks + size + len(current)):
                size = 1

            # chunked forward step, look for best players
            # to add to current team
            best = tuple()
            best_score = 0
            for chunk in divide(remainder, size):
                team = tuple(current + chunk)
                score = self._evaluator.play(team)
                results[team] = score.score
                # select best chunk
                if self._add == "chunk":
                    if score.score > best_score:
                        best = team
                        best_score = score.score
                # or select best player from each chunk
                else:
                    performances = score.performances
                    best = (*best, max(chunk, key=performances.get))
                # need to stop early
                if self.budget < 1:
                    break

            # don't have a score for the model yet
            if self._add == "best" and self.budget > 0:
                best_result = self._evaluator.play(best)
                best_score = best_result.score
                results[best] = best_result.score

            # backward step, iteratvely try to remove
            # features until the score stops getting better
            while len(best) > 1 and self.budget > 0:

                # get best feature to remove
                new_best = None
                new_best_score = 0
                for feature in best:
                    new_team = list(best)
                    new_team.remove(feature)
                    new_result = self._evaluator.play(new_team)
                    if new_result.score > new_best_score:
                        new_best = tuple(new_team)
                        new_best_score = new_result.score
                    # need to stop
                    if self.budget < 1:
                        break

                # only add the best one to the results dict
                results[new_best] = new_best_score

                # found a better one
                if new_best_score > best_score:
                    best = new_best
                    best_score = new_best_score

                # no better one
                else:
                    break

            # prepare for next one
            current = list(best)
            remainder = self.remainder(current)

        # store results
        self._results = results

    def remainder(self, features: Iterable[Hashable]) -> List[Hashable]:
        """Get remainder."""
        return list(set(self._df.columns) - set(features) - {self._target})


Individual = FrozenSet[Hashable]
Population = List[Individual]


class CHCGASelector(Selector):
    """Selector using GA.

    When warm started, the population is initialised to the
    starting genome and we begin with a cataclysmic mutation.

    """

    def __init__(
        self,
        evaluator: Game = None,
        games: int = 1000,
        population: int = 20,
        initial: int = 8,
    ):
        """

        Args:
            size: The size of the population.
            initial: The number of chromosomes
                in the initial population.

        """
        super().__init__(evaluator, games)
        self._size = population
        self._initial = initial

    def run(self):
        # initialise threshold
        self._threshold_base = len(self.chromosomes) // 4
        self._threshold = self._threshold_base
        self._results: Dict[Individual, float] = dict()
        # generate population
        population = self._generate_population()
        # perform iterations as long as there is budget
        while self.budget > 0:
            population = self.evolve(population)

    def _generate_population(self) -> Population:
        """Generate population."""
        if len(self._start) > 0:
            population = [frozenset(self._start) for _ in range(self._size)]
        else:
            shuffled = random.sample(self.chromosomes, k=len(self.chromosomes))
            population = [frozenset(d) for d in split(shuffled, self._size)]
        self.evaluate(population)
        return population

    def evaluate(self, individuals: Iterable[Individual]) -> None:
        """Evaluate indivduals."""
        for individual in individuals:
            # if individual not in self._results:
            if self.budget > 0:
                self._results[individual] = self._evaluator.play(individual).score
            else:
                self._results[individual] = 0

    def evolve(self, population: Population) -> Population:
        """Perform one iteration."""

        # sample current population without replacement by shuffling
        # and popping two by two
        shuffled = random.sample(population, len(population))

        # generate children
        children = list()
        while len(shuffled) > 0:
            p1 = shuffled.pop()
            p2 = shuffled.pop()
            offspring = self.cross(p1, p2)
            if offspring is not None:
                children.extend(offspring)

        # drop threshold if no offspring
        if len(children) == 0:
            self._threshold = self._threshold - 1
            # when threshold drops to 0 or stagnated, perform
            # cataclysmic mutation and rest the threshold
            if self._threshold == 0 or len(set(children)) == 1:
                best = max(population, key=self._results.get)
                children = [best]
                while len(children) < self._size:
                    children.append(self.mutate(best, 0.4))
                self._threshold = self._threshold_base

        # evaluate children
        self.evaluate(children)

        # elitist selection
        ranked = sorted(population + children, key=self._results.get, reverse=True)
        elites = ranked[: self._size]

        # replace current population
        return elites

    def cross(self, a: Individual, b: Individual) -> Tuple[Individual, Individual]:
        """Perform Half-Uniform Crossover (HUX).

        Crosses half of the non-matching alleles.

        Args:
            threshold: Minimal number of different alleles
                in order to allow crossing two individuals.

        Returns:
            If at least threshold alleles are different,
            return a tuple of offspring. Else, return None.

        """
        # get all alleles where a and b differ
        different = (a - b) | (b - a)
        # don't cross over
        if len(different) // 2 < self._threshold:
            return None
        # select chromosomes to swamp
        swap = np.random.choice(list(different), len(different) // 2, replace=False)
        # make children
        c1 = set(a)
        c2 = set(b)
        for chromosome in swap:
            for child in [c1, c2]:
                if chromosome in child:
                    child.remove(chromosome)
                else:
                    child.add(chromosome)
        # failed to make children
        if len(c1) < 2 or len(c2) < 2:
            return None
        return frozenset(c1), frozenset(c2)

    def mutate(self, i: Individual, p: float = 0.1) -> Individual:
        """Mutation.

        Args:
            i: Individual to mutate.
            p: Percentage of bits to flip.

        Returns:
            Individual with p chromosomes mutated.

        """
        n = len(self._df.columns)
        while True:
            mutate = np.random.choice(self.chromosomes, size=int(p * n), replace=False)
            genome = set(i)
            for chromosome in mutate:
                if chromosome in genome:
                    genome.remove(chromosome)
                else:
                    genome.add(chromosome)
            if len(genome) > 1:
                return frozenset(genome)

    @cached_property
    def chromosomes(self) -> List[Hashable]:
        return list(set(self._df.columns) - {self._target})


# class Population:
#     """Population of individuals."""

#     def __init__(
#         self, population: List["Individual"], fitness: Callable[["Individual"], float]
#     ):
#         """

#         Args:
#             population: Initial population.
#             fitness: Fitness function.

#         """
#         self._population = {
#             individual: fitness(individual.genome) for individual in population
#         }
#         self._n = len(self._population)
#         self._fitness = fitness
#         self._threshold_base = len(population[0]) // 4
#         self._threshold = self._threshold_base

#     def evolve(self):
#         """Evolve to next generation.

#         Args:
#             evaluate: Function that evaluates an individual.

#         """

#         # sample current population without replacement by shuffling
#         # and popping two by two.
#         shuffled = random.sample(list(self._population), len(self._population))

#         # generate children
#         children = list()
#         while len(shuffled) > 0:
#             p1 = shuffled.pop()
#             p2 = shuffled.pop()
#             offspring = p1.cross(p2, self._threshold)
#             if offspring is not None:
#                 children.extend(offspring)

#         # drop threshold if no offspring
#         if len(children) == 0:
#             self._threshold = self._threshold - 1

#             # threshold drops to 0, perform cataclysmic mutation
#             if self._threshold == 0 or self.has_stagnated():
#                 best = self.best()
#                 children = [best]
#                 while len(children) < self._n:
#                     children.append(best.mutate(0.4))
#                 # reset the threshold
#                 self._threshold = self._threshold_base

#         # collect all individuals of current iteration and
#         # evaluate them
#         combined = {child: self._fitness(child.genome) for child in children}
#         combined.update(self._population)

#         # elitist selection
#         ranked = sorted(combined, key=combined.get, reverse=True)
#         elites = ranked[: self._n]

#         # replace current population
#         self._population = {elite: combined[elite] for elite in elites}

#     def evolve_n(self, n: int):
#         """Evolve `n` times."""
#         for _ in range(n):
#             self.evolve()

#     def fitness(self, individual: "Individual"):
#         return self._population[individual]

#     def best(self) -> "Individual":
#         return max(self._population, key=self._population.get)

#     def individuals(self) -> List["Individual"]:
#         return list(self._population)

#     def has_stagnated(self) -> bool:
#         """Check if all elements are equal."""
#         iterator = iter(self._population)
#         try:
#             first = next(iterator)
#         except StopIteration:
#             return True
#         return all(np.equal(first.genome, rest.genome).all() for rest in iterator)

#     @property
#     def n(self):
#         return self._n

#     def __str__(self):
#         return "\n".join("{} {}".format(i, s) for i, s in self._population.items())


# class Individual:
#     def __init__(self, genome: Set[Hashable]):
#         self._genome = genome

#     def cross(
#         self, other: "Individual", threshold: int
#     ) -> Optional[Tuple["Individual"]]:
#         """Perform Half-Uniform Crossover (HUX).

#         Crosses half of the non-matching alleles.

#         Args:
#             threshold: Minimal number of different alleles
#                 in order to allow crossing two individuals.

#         Returns:
#             If at least threshold alleles are different,
#             return a tuple of offspring. Else, return None.

#         """
#         assert threshold < len(self.genome)
#         assert len(self.genome) == len(other.genome)
#         # get locations where differ
#         (different,) = np.where(self.genome != other.genome)
#         # don't cross over
#         if len(different) // 2 < threshold:
#             return None
#         # select chromosomes to swamp
#         indices = np.random.choice(different, len(different) // 2, replace=False)
#         # make children
#         c1 = np.copy(self.genome)
#         c1[indices] = other.genome[indices]
#         c2 = np.copy(other.genome)
#         c2[indices] = self.genome[indices]
#         if c1.sum() < 2 or c2.sum() < 2:
#             return None
#         return Individual(c1), Individual(c2)

#     def mutate(self, p: float) -> "Individual":
#         """Mutation.

#         Args:
#             p: Percentage of bits to flip.

#         """
#         assert p < 1
#         while True:
#             indices = np.random.choice(
#                 np.arange(len(self.genome)), int(p * len(self.genome)), replace=False
#             )
#             genome = np.copy(self.genome)
#             genome[indices] = 1 - genome[indices]
#             if genome.sum() > 1:
#                 return Individual(genome)

#     @property
#     def genome(self):
#         return self._genome

#     @classmethod
#     def generate(cls, n: int) -> "Individual":
#         """Generate random genome with.."""
#         while True:
#             candidate = np.random.randint(2, size=n)
#             if candidate.sum() > 1:
#                 return Individual(candidate)
#         # return Individual(np.random.randint(2, size=n))

#     def __eq__(self, other: "Individual"):
#         return id(self) == id(other)

#     def __len__(self):
#         return len(self._genome)

#     def __str__(self):
#         return str(self._genome)

#     def __hash__(self):
#         return hash(id(self))
