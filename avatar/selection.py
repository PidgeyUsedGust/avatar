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
from sklearn.tree import export_text
from mercs.core import Mercs
from .utilities import to_mercs, to_m_codes
from .analysis import FeatureEvaluator



class Selector:
    """Base feature selector."""

    def __init__(self, evaluator: FeatureEvaluator = None):
        """

        Args:
            iterations: Number of iterations.

        """

        if evaluator is None:
            evaluator = FeatureEvaluator(method=None, n_folds=4, max_depth=4)
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

    @abstractmethod
    def select(self) -> List[Label]:
        """Select features.
        
        Automatically select the optimal number of features according
        to this model.
    
        """
        pass

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
        iterations: int = 100,
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

        # initialise
        scores = np.zeros(self._iterations)
        fimps = np.zeros((self._iterations, len(self._df.columns)))
        masks = np.zeros((self._iterations, len(self._df.columns)))

        # run sampling
        for i in range(self._iterations):
            mask = self._generate_mask()
            masks[i] = mask
            scores[i], fimps[i] = self._evaluator.evaluate(mask)

        # scale with counts and re-normalise
        # print(masks.sum(axis=0))
        fimps = fimps / masks.sum(axis=0)
        fimps = fimps / fimps.sum(axis=1).reshape((-1, 1))

        self._fimps = fimps
        self._scores = scores

    def select(self) -> List[Label]:
        """Select features until the model is sufficiently explained."""
        scores = self.scores()
        ranked = np.argsort(scores).tolist()
        select = list()
        while np.sum(scores[select]) < self._explain:
            select.append(ranked.pop())
        return self._df.columns[select]

    def _generate_mask(self) -> np.ndarray:
        while True:
            mask = self._rng.randint(2, size=len(self._df.columns))
            if mask.sum() > 1:
                return mask


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
            self.add(best)
            # print("Adding", self._df.columns[best])
            k += 1
            if k not in self._best:
                self._best[k] = (self.mask(), score)
                best_score[k] = score
                best_fimps[k] = fimps

            # start conditional exclusion
            while k > 1:
                # backward step (SBS)
                best, (score, fimps) = self.backward()
                # best (k - 1) subset so far
                if score > best_score[k - 1]:
                    self.remove(best)
                    # print("Removing", self._df.columns[best])
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

    def select(self) -> List[Label]:
        select_from = [(score, k, mask) for k, (mask, score) in self._best.items()]
        select_from = sorted(select_from, key=lambda x: (-x[0], x[1]))
        select = select_from[0][2].astype(bool)
        return self._df.columns[select]

    def forward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform forward step."""
        to_add = set(np.where(self._mask == 0)[0]) - {self._target_i}
        scores = {f: self._evaluator.evaluate(self.mask(add=f)) for f in to_add}
        best = max(scores, key=lambda s: scores[s][0])
        return best, scores[best]

    def backward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform backward step."""
        to_remove = set(np.where(self._mask == 1)[0]) - {self._target_i}
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


class CHCGASelector(Selector):
    """Selector using GA."""

    def __init__(
        self,
        population_size: int = 40,
        iterations: int = 50,
        evaluator: Optional[FeatureEvaluator] = None,
    ):
        super().__init__(evaluator)
        self._population_size = population_size
        self._iterations = iterations

    def run(self):
        # generate population
        population = Population.generate_for(
            self._df, n=self._population_size, fitness=self._evaluator.accuracy
        )
        # evolve `iterations` times
        for _ in range(self._iterations):
            population.evolve()
        # get importances
        importances = np.zeros((self._population_size, len(self._df.columns)))
        scores = np.zeros(self._population_size)
        for i, individual in enumerate(population.individuals):
            importances[i] = self._evaluator.importances(individual.genome)
            scores[i] = population.fitness(individual)
        # aggregate
        self._fimps = (importances * scores.reshape((-1, 1))).mean(axis=0)
        self._fimps = self._fimps / self._fimps.sum()
        self._scores = scores


class Population:
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
        self._threshold = len(population[0]) // 4

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
            if self._threshold == 0:
                best = self.best()
                children = [best]
                while len(children) < self._n:
                    children.append(best.mutate(0.4))

        # collect all individuals of current iteration and
        # evaluate them
        combined = {child: self._fitness(child.genome) for child in children}
        combined.update(self._population)

        # elitist selection
        ranked = sorted(combined, key=combined.get, reverse=True)
        elites = ranked[: self._n]

        # replace current population
        self._population = {elite: combined[elite] for elite in elites}

    def evaluate(self, individual: "Individual") -> Dict["Individual", float]:
        return {individual: evaluator(individual) for individual in self._population}

    def best(self) -> "Individual":
        return max(self._population, key=self._population.get)

    def average(self) -> float:
        return np.mean(list(self._population.values()))

    def fitness(self, individual: "Individual") -> float:
        return self._population[individual]

    @property
    def individuals(self):
        return self._population

    @property
    def n(self):
        return self._n

    @classmethod
    def generate_for(self, df: pd.DataFrame, n: int, fitness) -> "Population":
        return Population([Individual.generate_for(df) for _ in range(n)], fitness)

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
        return Individual(c1), Individual(c2)

    def mutate(self, p: float) -> "Individual":
        """Mutation.
        
        Args:
            p: Percentage of bits to flip.
    
        """
        assert p < 1
        indices = np.random.choice(
            np.arange(len(self.genome)), int(p * len(self.genome)), replace=False
        )
        genome = np.copy(self.genome)
        genome[indices] = 1 - genome[indices]
        return Individual(genome)

    @property
    def genome(self):
        return self._genome

    @classmethod
    def generate(cls, n: int) -> "Individual":
        """Generate random genome."""
        return Individual(np.random.randint(2, size=n))

    @classmethod
    def generate_for(cls, df: pd.DataFrame) -> "Individual":
        """Generate for a dataframe."""
        return cls.generate(df.shape[1])

    def __len__(self):
        return len(self._genome)

    def __str__(self):
        return str(self._genome)

    def __hash__(self):
        return hash(self._genome.tobytes())
