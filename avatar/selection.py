"""Feature selection.

Two types of selection are performed.

 * Filters will use intrinsic properties of a single column
   in order to quickly remove columns for consideration.

"""
import random
import numpy as np
import pandas as pd
from abc import abstractmethod, ABC
from typing import Union, Optional, Set, Tuple, List, Callable, Dict
from pandas._typing import Label
from pandas.api.types import is_numeric_dtype  # , is_string_type
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.tree import export_text
from mercs.core import Mercs
from .utilities import to_mercs, to_m_codes
from .analysis import FeatureEvaluator
from .settings import verbose


np.random.seed(1337)
random.seed(1337)


class Selector:
    """Base feature selector."""

    def __init__(self, evaluator: FeatureEvaluator = None):
        """

        Args:
            explain: Percentage of feature importance to select. If
                set to 0, decide automatically.

        """

        if evaluator is None:
            evaluator = FeatureEvaluator(n_folds=4, max_depth=4, n_samples=1000)
        self._evaluator = evaluator

        # the following attributes are to be filled by the
        # selector implementations
        self._fimps = None
        self._scores = None

    def fit(
        self,
        df: pd.DataFrame,
        target: Optional[Label] = None,
        start: Optional[List[Label]] = None,
    ):
        """Fit selector.

        Args:
            start: Initial list of features to include. By default, don't
                include any features.

        """
        self._df = df
        self._target = target
        self._target_i = df.columns.get_loc(target)
        self._start = start if start is not None else list()
        self._evaluator.fit(df, target)
        self.run()

    @abstractmethod
    def run(self):
        """Rank features.

        This method should fill `self._fimps` and `self._scores`
        with a matrix of feature importances and the associated
        score of those models on some other data.

        """
        pass

    def select(self, explain: float = 0.9, max_features: int = 0) -> List[Label]:
        """Select features until the model is sufficiently explained."""
        scores = self.scores()
        ranked = np.argsort(scores)[::-1]
        select = np.searchsorted(np.cumsum(scores[ranked]), explain)
        select = min(select, max_features or len(self._df.columns))
        return self._df.columns[ranked[:select]].tolist()

    def scores(self) -> np.ndarray:
        """Get feature scores."""
        scores = self._fimps * self._scores.reshape((-1, 1))
        scores = scores.sum(axis=0) / scores.sum()
        return scores

    def ranked(self) -> List[Label]:
        """Rank all features."""
        return self._df.columns[np.argsort(self.scores())[::-1]]

    def best(self, n: int) -> List[Label]:
        """Select best `n` features."""
        return self.ordered()[:n]

    def _start_mask(self) -> np.ndarray:
        """Generate mask with starting features."""
        mask = np.zeros(len(self._df.columns))
        mask[[self._df.columns.get_loc(c) for c in self._start]] = 1
        # mask[self._df.columns.get_loc(self._target)] = 1
        return mask

    def __str__(self) -> str:
        s = ""
        for i in np.argsort(self._fimps)[::-1]:
            s += "{}  {}\n".format(self._df.columns[i], self._fimps[i])
        return s


class SamplingSelector(Selector):
    """Randomly sample sets of features.

    This selector ignores the start, but a variation of random
    sampling with warm start is implemented in `WarmSamplingSelector`.

    """

    def __init__(
        self,
        iterations: Union[int, float] = 1600,
        explain: float = 0.9,
        evaluator: Optional[FeatureEvaluator] = None,
    ):
        """

        Args:
            iterations: Number of iterations to run for.
            explain: Percentage of model to explain for automatic
                number of feature selection.

        """
        super().__init__(evaluator)
        self._iterations = iterations
        self._explain = explain
        self._rng = np.random.RandomState(1337)

    def run(self):
        """Generate random sets of features and evaluate them."""
        if isinstance(self._iterations, float):
            iterations = int(len(self._df.columns) * self._iterations)
        else:
            iterations = self._iterations
        # initialise
        scores = np.zeros(iterations)
        fimps = np.zeros((iterations, len(self._df.columns)))
        masks = np.zeros((iterations, len(self._df.columns)))
        # run sampling
        for i in tqdm(range(iterations), desc="Ranking features", disable=not verbose):
            mask = self._generate_mask()
            masks[i] = mask
            scores[i], fimps[i] = self._evaluator.evaluate(mask)
        # scale with counts and re-normalise
        fimps = fimps / masks.sum(axis=0)
        fimps = fimps / fimps.sum(axis=1).reshape((-1, 1))
        self._fimps = fimps
        self._scores = scores

    def _generate_mask(self) -> np.ndarray:
        while True:
            mask = self._rng.randint(2, size=len(self._df.columns))
            if mask.sum() > 1:
                return mask

    def __str__(self) -> str:
        return "SamplingSelector(iterations={}, evaluator={})".format(
            self._iterations, self._evaluator
        )


class WarmSamplingSelector(SamplingSelector):
    """Sampling selector with hot start.

    Starts with the base mask and swaps every bit with a
    predefined probability.

    """

    def _generate_mask(self) -> np.ndarray:
        mask = self._rng.random(size=len(self._df.columns)) < (1 / 3)
        base = self._start_mask()
        base[mask] = 1 - base[mask]
        return base


class SFFSelector(Selector):
    """Sequential Forward Floating Selection."""

    def __init__(
        self, iterations: int = 50, evaluator: Optional[FeatureEvaluator] = None
    ):
        super().__init__(evaluator)
        self._iterations = iterations
        self._mask = None
        self._best = {}

    def run(self):
        """Perform SFFS."""

        best_score = np.zeros(len(self._df.columns))
        best_fimps = np.zeros((len(self._df.columns), len(self._df.columns)))

        # reset mask
        self._mask = self._start_mask()
        k = int(self._mask.sum())

        for _ in range(self._iterations):

            # forward step (SFS)
            best, (score, fimps) = self.forward()
            if best >= 0:
                self.add(best)
                k += 1
                if k not in self._best:
                    self._best[k] = (self.mask(), score)
                    best_score[k] = score
                    best_fimps[k] = fimps

            # start conditional exclusion
            while k > 1:
                # backward step (SBS)
                best, (score, fimps) = self.backward()
                if best >= 0:
                    # best (k - 1) subset so far
                    if score > best_score[k - 1]:
                        self.remove(best)
                        self._best[k - 1] = (self.mask(), score)
                        best_score[k - 1] = score
                        best_fimps[k - 1] = fimps
                        k -= 1
                    # back to start of algorithm
                    else:
                        break

        # count number of non-zero k rows
        self._fimps = best_fimps
        self._scores = best_score

    def select(self, explain: float = 0.9, max_features: int = 0) -> List[Label]:
        max_features = max_features or len(self._df.columns)
        score, k, mask = max(
            [
                (score, k, mask)
                for k, (mask, score) in self._best.items()
                if k <= max_features
            ],
            key=lambda x: (x[0], -x[1]),
        )
        actual = np.where(self._fimps[k] > 0)[0]
        return self._df.columns[actual].tolist()

    def forward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform forward step."""
        to_add = set(np.where(self._mask == 0)[0]) - {self._target_i}
        if len(to_add) == 0:
            return -1, (0, np.array([]))
        scores = {f: self._evaluator.evaluate(self.mask(add=f)) for f in to_add}
        best = max(scores, key=lambda s: scores[s][0])
        return best, scores[best]

    def backward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform backward step."""
        to_remove = set(np.where(self._mask == 1)[0]) - {self._target_i}
        if len(to_remove) == 0:
            return -1, (0, np.array([]))
        scores = {f: self._evaluator.evaluate(self.mask(remove=f)) for f in to_remove}
        best = max(scores, key=lambda s: scores[s][0])
        return best, scores[best]

    def add(self, feature: int):
        self._mask[feature] = 1

    def remove(self, feature: int):
        self._mask[feature] = 0

    def mask(self, add=None, remove=None):
        """Get mask with one update."""
        mask = np.copy(self._mask)
        if add is not None:
            mask[add] = 1
        if remove is not None:
            mask[remove] = 0
        return mask

    def __str__(self) -> str:
        return "SFFSSelector(iterations={}, evaluator={})".format(
            self._iterations, self._evaluator
        )


class CHCGASelector(Selector):
    """Selector using GA.

    When warm started, the population is initialised to the
    starting genome and we begin with a cataclysmic mutation.

    """

    def __init__(
        self,
        population_size: int = 40,
        iterations: int = 50,
        explain: float = 0.9,
        evaluator: Optional[FeatureEvaluator] = None,
    ):
        super().__init__(evaluator)
        self._population_size = population_size
        self._iterations = iterations
        self._explain = explain

    def run(self):
        # generate population
        population = self._generate_population()
        # evolve `iterations` times
        population.evolve_n(self._iterations)
        # get importances
        importances = np.zeros((self._population_size, len(self._df.columns)))
        scores = np.zeros(self._population_size)
        for i, individual in enumerate(population.individuals()):
            importances[i] = self._evaluator.importances(individual.genome)
            scores[i] = population.fitness(individual)
        # store
        self._fimps = importances
        self._scores = scores

    def _generate_population(self) -> "Population":
        """Generate population."""
        n = self._population_size
        s = len(self._df.columns)
        # start values provided
        if len(self._start) > 0:
            individuals = [Individual(self._start_mask()) for _ in range(n)]
        # nothing provided, start from scratch
        else:
            individuals = [Individual.generate(s) for _ in range(n)]
        return Population(individuals, fitness=self._evaluator.accuracy)

    def __str__(self) -> str:
        return "CHCGASelector(iterations={}, population={}, evaluator={})".format(
            self._iterations, self._population_size, self._evaluator
        )


class Population:
    """Population of individuals."""

    def __init__(
        self, population: List["Individual"], fitness: Callable[["Individual"], float]
    ):
        """

        Args:
            population: Initial population.
            fitness: Fitness function.

        """
        self._population = {
            individual: fitness(individual.genome) for individual in population
        }
        self._n = len(self._population)
        self._fitness = fitness
        self._threshold_base = len(population[0]) // 4
        self._threshold = self._threshold_base

    def evolve(self):
        """Evolve to next generation.

        Args:
            evaluate: Function that evaluates an individual.

        """

        # sample current population without replacement by shuffling
        # and popping two by two.
        shuffled = random.sample(list(self._population), len(self._population))

        # generate children
        children = list()
        while len(shuffled) > 0:
            p1 = shuffled.pop()
            p2 = shuffled.pop()
            offspring = p1.cross(p2, self._threshold)
            if offspring is not None:
                children.extend(offspring)

        # drop threshold if no offspring
        if len(children) == 0:
            self._threshold = self._threshold - 1

            # threshold drops to 0, perform cataclysmic mutation
            if self._threshold == 0 or self.has_stagnated():
                best = self.best()
                children = [best]
                while len(children) < self._n:
                    children.append(best.mutate(0.4))
                # reset the threshold
                self._threshold = self._threshold_base

        # collect all individuals of current iteration and
        # evaluate them
        combined = {child: self._fitness(child.genome) for child in children}
        combined.update(self._population)

        # elitist selection
        ranked = sorted(combined, key=combined.get, reverse=True)
        elites = ranked[: self._n]

        # replace current population
        self._population = {elite: combined[elite] for elite in elites}

    def evolve_n(self, n: int):
        """Evolve `n` times."""
        for _ in range(n):
            self.evolve()

    def fitness(self, individual: "Individual"):
        return self._population[individual]

    def best(self) -> "Individual":
        return max(self._population, key=self._population.get)

    def individuals(self) -> List["Individual"]:
        return list(self._population)

    def has_stagnated(self) -> bool:
        """Check if all elements are equal."""
        iterator = iter(self._population)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(np.equal(first.genome, rest.genome).all() for rest in iterator)

    @property
    def n(self):
        return self._n

    def __str__(self):
        return "\n".join("{} {}".format(i, s) for i, s in self._population.items())


class Individual:
    def __init__(self, genome: np.ndarray):
        self._genome = genome

    def cross(
        self, other: "Individual", threshold: int
    ) -> Optional[Tuple["Individual"]]:
        """Perform Half-Uniform Crossover (HUX).

        Crosses half of the non-matching alleles.

        Args:
            threshold: Minimal number of different alleles
                in order to allow crossing two individuals.

        Returns:
            If at least threshold alleles are different,
            return a tuple of offspring. Else, return None.

        """
        assert threshold < len(self.genome)
        assert len(self.genome) == len(other.genome)
        # get locations where differ
        (different,) = np.where(self.genome != other.genome)
        # don't cross over
        if len(different) // 2 < threshold:
            return None
        # select chromosomes to swamp
        indices = np.random.choice(different, len(different) // 2, replace=False)
        # make children
        c1 = np.copy(self.genome)
        c1[indices] = other.genome[indices]
        c2 = np.copy(other.genome)
        c2[indices] = self.genome[indices]
        if c1.sum() < 2 or c2.sum() < 2:
            return None
        return Individual(c1), Individual(c2)

    def mutate(self, p: float) -> "Individual":
        """Mutation.

        Args:
            p: Percentage of bits to flip.

        """
        assert p < 1
        while True:
            indices = np.random.choice(
                np.arange(len(self.genome)), int(p * len(self.genome)), replace=False
            )
            genome = np.copy(self.genome)
            genome[indices] = 1 - genome[indices]
            if genome.sum() > 1:
                return Individual(genome)

    @property
    def genome(self):
        return self._genome

    @classmethod
    def generate(cls, n: int) -> "Individual":
        """Generate random genome with.."""
        while True:
            candidate = np.random.randint(2, size=n)
            if candidate.sum() > 1:
                return Individual(candidate)
        # return Individual(np.random.randint(2, size=n))

    def __eq__(self, other: "Individual"):
        return id(self) == id(other)

    def __len__(self):
        return len(self._genome)

    def __str__(self):
        return str(self._genome)

    def __hash__(self):
        return hash(id(self))
