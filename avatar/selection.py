"""Feature selection.

Two types of selection are performed.

 * Preselectors will use intrinsic properties of a single column
   in order to quickly remove columns for consideration.

"""
import random
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Union, Optional, Set, Tuple, List, Callable, Dict
from pandas._typing import Label
from pandas.api.types import is_numeric_dtype  # , is_string_type
from mercs.core import Mercs
from .utilities import to_mercs, to_m_codes
from .analysis import FeatureEvaluator


class StackedPreselector:
    """Combine differenct selectors."""

    def __init__(self, selectors):
        self._selectors = selectors

    def select(self, df: pd.DataFrame):
        for selector in self._selectors:
            df = selector.select(df)
        return df


class MissingPreselector:
    """Remove columns missing at least a percentage of values."""

    def __init__(self, threshold: float = 0.5):
        self._threshold = 0.5

    def select(self, df: pd.DataFrame):
        return df.dropna(axis=1, thresh=(self._threshold * len(df.index)))


class ConstantPreselector:
    """Remove columns with constants."""

    def select(self, df: pd.DataFrame):
        return df.loc[:, (df != df.iloc[0]).any()]


class IdenticalPreselector:
    """Remove identical columns.
    
    Remove numerical columns that are identical.

    """

    def select(self, df: pd.DataFrame):
        return df.drop(self.duplicates(df), axis=1)

    def duplicates(self, df: pd.DataFrame):
        duplicates = set()
        for i in range(df.shape[1]):
            col_one = df.iloc[:, i]
            for j in range(i + 1, df.shape[1]):
                col_two = df.iloc[:, j]
                if col_one.equals(col_two):
                    duplicates.add(df.columns[j])
        return duplicates


class BijectivePreselector:
    """Remove categorical columns that are a bijection."""

    def select(self, df: pd.DataFrame):
        return df.drop(self.duplicates(df), axis=1)

    def duplicates(self, df: pd.DataFrame):
        # select only object
        df = df.select_dtypes(include="object")
        duplicates = set()
        for i in range(df.shape[1]):
            col_one = df.iloc[:, i]
            for j in range(i + 1, df.shape[1]):
                col_two = df.iloc[:, j]
                if (col_one.factorize()[0] == col_two.factorize()[0]).all():
                    duplicates.add(df.columns[j])
        return duplicates


class UniquePreselector:
    """Remove columns containing only categorical, unique elements."""

    def select(self, df: pd.DataFrame):
        uniques = list()
        for column in df:
            if df[column].dtype.name in ["object", "category"]:
                if df[column].dropna().is_unique:
                    uniques.append(column)
        return df.drop(uniques, axis=1)


class IterativePreselector:
    """Train decision stump for every feature individually."""

    def __init__(self, threshold: float = 0.95):
        """
        
        Args:
            threshold: Features that make the same prediction for
                `threshold` percentage of rows are discarded.

        """
        self.threshold = 1 - threshold

    def predictions(self, df: pd.DataFrame, target) -> pd.DataFrame:
        """Make predictions.
        
        Returns:
            Dataframe with same shape as `df` containing predictions for
            every feature.
            
        """

        # prepare data for mercs
        data, nominal = to_mercs(df)
        data = data.values
        data_test = np.nan_to_num(data)

        # initialise mask
        base_m_code = to_m_codes(df.columns, target)
        base_m_code[base_m_code == 0] = -1

        # perform predictions
        predictions = pd.DataFrame(0, index=df.index, columns=df.columns)
        for i, column in enumerate(df.columns):
            if column == target:
                continue
            m_code = np.copy(base_m_code)
            m_code[:, i] = 0
            model = Mercs(classifier_algorithm="DT", max_depth=1)
            model.fit(data, nominal_attributes=nominal, m_codes=m_code)
            predictions[column] = model.predict(data_test, q_code=m_code[0])

        return predictions

    def select(self, df: pd.DataFrame, target) -> pd.DataFrame:
        """Perform selection.
        
        Returns:
            A dataframe containing only selected features.
    
        """

        # get predictions
        predictions = self.predictions(df, target)

        # get columns similar to another columns
        similar = set()
        for i, ci in enumerate(predictions.columns):
            if ci in similar:
                break
            column = predictions[ci]
            column_type = is_numeric_dtype(df[ci])
            for cj in predictions.columns[i + 1 :]:
                if cj in similar:
                    break
                other = predictions[cj]
                other_type = is_numeric_dtype(df[cj])
                if (column_type and other_type) or (not column_type and not other_type):
                    d = np.abs(column - other).sum() / len(column)
                    if d < self.threshold:
                        similar.add(cj)

        # similar ones and return
        return df.drop(similar, axis=1)


class Selector:
    """Base feature selector."""

    def __init__(self, df: pd.DataFrame, target: Optional[Label] = None):
        self._df = df
        self._target = target
        self._evaluator = FeatureEvaluator(df, target, method=None, max_depth=2)
        # the following attributes are to be filled by the
        # selector implementations
        self._fimps = None
        self._scores = None

    @abstractmethod
    def run(self):
        """Rank features."""
        pass

    def scores(self) -> Dict[Label, float]:
        """Get scores."""
        scores = self._fimps * self._scores.reshape((-1, 1))
        scores = scores.sum(axis=0) / scores.sum()

    def ordered(self) -> List[Label]:
        return self._df.columns[np.argsort(self._importances)[::-1]]

    def select(self, n: int) -> List[Label]:
        return self._df.columns[self.ordered()[:n]]

    def __str__(self) -> str:
        s = ""
        for i in np.argsort(self._importances)[::-1]:
            s += "{}  {}\n".format(self._df.columns[i], self._importances[i])
        return s


class SamplingSelector(Selector):
    """Randomly sample sets of features and obtain feature relevances."""

    def __init__(self, df: pd.DataFrame, target: Label):
        super().__init__(df, target)
        self._rng = np.random.RandomState(1336)

    def run(self, iterations: int = 20):
        """Generate random sets of features and evaluate them."""
        self._fimps = np.zeros((iterations, len(self._df.columns)))
        self._scores = np.zeros(iterations)
        counts = np.zeros(len(self._df.columns))
        for i in range(iterations):
            mask = self.generate_mask()
            self._scores[i], self._fimps[i] = self._evaluator.evaluate(mask)
            counts = counts + mask
        # scale with counts and re-normalise
        self._fimps = self._fimps / counts
        self._fimps = self._fimps / self._fimps.sum(axis=1).reshape((-1, 1))

    def generate_mask(self) -> np.ndarray:
        return self._rng.randint(2, size=len(self._df.columns))


class SFFSelector(Selector):
    """Sequential Forward Floating Selection."""

    def __init__(self, df: pd.DataFrame, target: Label):
        super().__init__(df, target)
        # initialise the mask to exclude everything
        self._mask = to_m_codes(df.columns, target=target)[0]
        self._mask[self._mask == 0] = -1
        # mappings of number of features to best set found
        # with that number
        self._best = {}
        self._best_score = np.zeros(len(df.columns))
        self._best_fimps = np.zeros((len(df.columns), len(df.columns)))

    def run(self, iterations=50):
        """Perform SFFS."""
        k = 0
        for _ in range(iterations):
            # forward step (SFS)
            best, (score, fimps) = self.forward()
            self.add(best)
            k += 1
            if k not in self._best:
                self._best[k] = self.mask()
                self._best_score[k] = score
                self._best_fimps[k] = fimps
            # don't perform backward if only one feature to remove
            if k < 2:
                continue
            # start conditional exclusion
            while True:
                # backward step (SBS)
                best, (score, fimps) = self.backward()
                # best (k - 1) subset so far
                if score > self._best_score[k - 1]:
                    self.remove(best)
                    self._best[k - 1] = self.mask()
                    self._best_score[k - 1] = score
                    self._best_fimps[k - 1] = fimps
                    k -= 1
                # back to start of algorithm
                else:
                    break
        # count number of non-zero k rows
        self._fimps = self._best_fimps
        self._scores = self._best_score

    def forward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform forward step."""
        (to_add,) = np.where(self._mask == -1)
        scores = {f: self._evaluator.evaluate(self.mask(add=f)) for f in to_add}
        best = max(scores, key=lambda s: scores[s][0])
        return best, scores[best]

    def backward(self) -> Tuple[int, Tuple[float, np.ndarray]]:
        """Perform backward step."""
        (to_remove,) = np.where(self._mask == 0)
        scores = {f: self._evaluator.evaluate(self.mask(remove=f)) for f in to_remove}
        best = max(scores, key=lambda s: scores[s][0])
        return best, scores[best]

    def add(self, feature):
        self._mask[feature] = 0

    def remove(self, feature):
        self._mask[feature] = -1

    def mask(self, add=None, remove=None):
        """Get mask with one update."""
        mask = np.copy(self._mask)
        if add is not None:
            mask[add] = 0
        if remove is not None:
            mask[remove] = -1
        return mask


class CHCGASelector(Selector):
    """Selector using GA."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[Label] = None,
        population_size: int = 40,
    ):
        super().__init__(df, target)
        self._population_size = population_size

    def run(self, iterations: int = 20):
        # generate population
        population = Population.generate_for(
            self._df, n=self._population_size, fitness=self._evaluator.accuracy
        )
        # evolve `iterations` times
        for _ in range(iterations):
            population.evolve()
        # get importances
        importances = np.zeros((self._population_size, len(self._df.columns)))
        scores = np.zeros(self._population_size)
        for i, individual in enumerate(population.individuals):
            importances[i] = self._evaluator.importances(individual.genome)
            scores[i] = population.fitness(individual)
        # aggregate
        self._importances = (importances * scores.reshape((-1, 1))).mean(axis=0)
        self._importances = self._importances / self._importances.sum()


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
                    children.append(best.mutate())

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
