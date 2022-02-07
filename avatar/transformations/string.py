"""String transformations."""
import re
import string
import itertools
import numpy as np
import pandas as pd
from typing import Counter, Tuple, List, Set
from collections import defaultdict, Counter
from operator import itemgetter

from pandas.core.tools.numeric import to_numeric
from .base import Transformation


class StringTransformation(Transformation):
    """Base string transformation."""

    allowed = ["string"]


class Split(StringTransformation):
    """Split column by delimiter."""

    max_difference = 20
    """Max difference between number of elements."""

    min_rows = 0.5
    """Minimal proportion of rows that should have the splitter."""

    def __init__(self, delimiter: str):
        self.delimiter = re.escape(delimiter)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        split = column.str.split(pat=self.delimiter, expand=True)
        for column in split:
            if split[column].nunique() == 2:
                split[column] = split[column].astype("category")
            if split[column].str.isnumeric().all():
                split[column] = pd.to_numeric(split[column])
        return split

    def __str__(self) -> str:
        # return "Split({})".format(re.sub(r"\\(.)", r"\1", self.delimiter))
        return "Split({})".format(self.delimiter)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Get possible delimiters.

        Get all substrings of non-alphanumeric characters.

        """
        # figure out all non alphanumeric strings and deduplicate them.
        split = column.str.split(pat=r"[a-zA-Z0-9]+").dropna()
        # test if maximal difference not exceeded
        split_sizes = split.apply(len)
        if split_sizes.max() - split_sizes.min() > cls.max_difference:
            return []
        # count arguments
        candidates = Counter()
        for delimiters in split:
            delimiters = {d for d in delimiters if d}
            for delimiter in delimiters:
                candidates[delimiter] += 1
        arguments = list()
        threshold = cls.min_rows * column.notna().sum()
        for candidate, count in candidates.most_common():
            if count > threshold:
                arguments.append((candidate,))
        return arguments


class SplitDummies(StringTransformation):
    """Split by delimiter and align by column.

    For example, a column

        A,B
        B,
        A,C

    gets split into

        A B
          B
        A   C

    which allows every token to be considered as a feature.

    """

    max_categories = 20

    def __init__(self, delimiter: str):
        self.delimiter = (
            delimiter.replace("(", "\(").replace(")", "\)").replace("+", "\+")
        )

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.get_dummies(sep=self.delimiter)

    def __str__(self) -> str:
        return "SplitAlign({})".format(self.delimiter)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Get possible delimiters.

        Similar to regular split, but filter on number of unique values.

        """

        # compute max number of categories
        if cls.max_categories <= 1:
            max_categories = len(column) * cls.max_categories
        else:
            max_categories = cls.max_categories

        # filter arguments that yield more than the allowed number
        # of categories
        arguments_all = Split.arguments(column)
        arguments = list()
        for (argument,) in arguments_all:
            n_categories = np.unique(
                column.str.split(pat=re.escape(argument), expand=True).fillna("").values
            ).shape[0]
            if n_categories < max_categories:
                arguments.append((argument,))
        return arguments


class ExtractNumberPattern(StringTransformation):
    """Extract a number from text.

    Can either extract any number or a fixed pattern.

    Regex for extracting numbers was taken from https://stackoverflow.com/a/4703508
    where \. was replaced with [.,] to also match commas.

    """

    default = r"([-+]?(?:(?:\d*[.,]\d+)|(?:\d+[.,]?))(?:[Ee][+-]?\d+)?)"
    table = {ord(value): "0" for value in string.digits}

    def __init__(self, pattern: str = ""):
        if pattern == "":
            self.pattern = self.default
        else:
            self.pattern = pattern

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        try:
            expanded = column.str.extract(pat=self.pattern, expand=True)
            expanded = pd.to_numeric(
                expanded.iloc[:, 0].str.replace(r",", "", regex=True)
            ).to_frame()
        except ValueError:
            expanded = pd.DataFrame()
        return expanded

    def __str__(self) -> str:
        return "ExtractNumberPattern({})".format(self.pattern[8:-8])

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        # get reduced representation
        column = column.dropna().map(cls.translate).drop_duplicates()
        # get all numbers
        numbers = column.str.extractall(pat=cls.default)
        # generate candidate patterns
        candidates = {cls.default}
        candidates.update(cls.pattern(number) for number in np.unique(numbers))
        candidates.update(re.sub(r"\{\d+\}", "+", c) for c in set(candidates))
        return [(pattern,) for pattern in candidates]

    @classmethod
    def translate(cls, string: str) -> str:
        return string.translate(cls.table)

    @classmethod
    def pattern(cls, string: str) -> List[str]:
        pattern = ""
        for k, g in itertools.groupby(string):
            if k == "0":
                pattern += r"\d{{{}}}".format(len(list(g)))
            else:
                pattern += re.escape(k)
        return r"(?:^|\D)(" + pattern + r")(?:\D|$)"


class ExtractInteger(StringTransformation):
    """Extract k'th simple number."""

    _cache = dict()
    """Cache for numbers."""

    def __init__(self, k: int = 0):
        self.k = k

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        if column.name not in self._cache:
            self._cache[column.name] = column.str.replace(
                r"^[^0-9]+", "", regex=True
            ).str.split(r"[^0-9]+", expand=True)
        return pd.to_numeric(self._cache[column.name].iloc[:, self.k]).to_frame()

    def __str__(self) -> str:
        return "ExtractInteger({})".format(self.k)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[int]]:
        # get reduced representation
        column = column.dropna().drop_duplicates()
        # get all numbers and counts
        numbers = column.str.split(r"[^0-9]+")
        counts = numbers.map(len)
        # check for empty value in beginning and end
        best = counts.max()
        penalty = best
        for row in numbers[counts == best]:
            if row[0] != "" and row[-1] != "":
                return [(k,) for k in range(best)]
            penalty = min(penalty, int(row[0] == "") + int(row[-1] == ""))
        return [(k,) for k in range(best - penalty)]


class ExtractFloat(StringTransformation):
    """Extract k'th float."""

    def __init__(self, k: int = 0):
        self.k = k

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        expanded = (
            column.str.extractall(pat=r"(\d+)", regex=True)
            .unstack()
            .droplevel(0, axis=1)
        )
        expanded = pd.to_numeric(expanded.iloc[:, self.k]).to_frame()
        return expanded

    def __str__(self) -> str:
        return "ExtractFloat({})".format(self.k)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[int]]:
        # get reduced representation
        column = column.dropna().drop_duplicates()
        # get all numbers
        numbers = column.str.split(r"[^0-9]+").apply(len)
        # turn into k
        return [(k,) for k in range(len(numbers.columns))]


class ExtractBoolean(StringTransformation):
    """Check whether a single word occurs."""

    occurences = 0.02
    """Minimal proportion of rows that has to contain the word."""

    _cache = dict()
    """Cache for storing expanded columns."""

    def __init__(self, word: str):
        self.word = word
        self._regex = re.compile("[^a-zA-Z]{}[^a-zA-Z]".format(word))

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        if column.name not in self._cache:
            self._cache[column.name] = (
                column.str.split(pat=r"[^a-zA-Z0-9]+").fillna("").map(set)
            )
        return self._cache[column.name].apply(lambda x: self.word in x).to_frame()

    def __str__(self) -> str:
        return "ExtractBoolean({})".format(self.word)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Extract words that occur in at least some rows."""
        n = len(column) * cls.occurences
        words = column.str.split(pat=r"[^a-zA-Z0-9]+").dropna().apply(set)
        counts = Counter(itertools.chain.from_iterable(words))
        arguments = set()
        for word, count in counts.most_common():
            if word and count > n:
                arguments.add((word,))
        return arguments


""" EXPERIMENTAL.

These are possible future transformations that need some work.

"""


class ExtractPattern(StringTransformation):
    """Extract a regex pattern from a string.

    Attributes:
        table: Translation table for classes. By default, considers
            lowercase, uppercase, digits and whitespace as character
            classes.
        max_length: Maximal length of regex patterns to be extacted.

    """

    table = {
        ord(value): key
        for values, key in {
            string.ascii_lowercase: "a",
            string.ascii_uppercase: "A",
            string.digits: "0",
            string.whitespace: " ",
        }.items()
        for value in values
    }

    max_length = 3

    def __init__(self, pattern: str):
        self._pattern = pattern

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.extract(pat=self._pattern, expand=True)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        translated = column.map(cls.translate).drop_duplicates()

    @classmethod
    def translate(cls, string: str) -> List[str]:
        return string.translate(cls.table)

    @classmethod
    def patterns(cls, string: str) -> List[str]:
        patterns = list()
        for character, group in itertools.groupby(string):
            n = len(list(group))
            patterns.append("{}{{{}}}".format(character, n))
            patterns.append("{}+".format(character))
        return patterns


""" DEPRECATED. """


class ExtractWord(StringTransformation):
    """Extract a word from a string."""

    beam = 1
    stop = 2
    max_words = 20

    def __init__(self, words: Set[str]):
        self.words = words
        self._regex = "({})".format("|".join(words))

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.extract(self._regex, expand=True)

    def __str__(self) -> str:
        return "ExtractWord([{}])".format(", ".join(sorted(self.words)))

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Set[str]]]:
        """Extract sets of words.

        Greedily pick smallest sets of words such that every
        row has exactly one of them.

        """
        result = list()
        # get sets of words, count occurrences and make into sets.
        words = column.str.extractall(pat=r"([a-zA-Z]+)")
        words = words.groupby(level=0)[0].apply(frozenset).to_list()
        count = Counter(list(itertools.chain.from_iterable(words)))
        # queue with counter and words
        queue = [(count, words, set())]
        while len(queue) > 0:
            counter, column, found = queue.pop()
            # reached max words
            if len(found) > cls.max_words:
                continue
            # if counter is empty, add found to result
            if len(counter) == 0:
                result.append((found,))
            for word, count in counter.most_common(cls.beam):
                # check if need t ostop
                if count <= cls.stop:
                    if len(found) > 0:
                        result.append((found,))
                    break
                # remove counts
                new_found = found | {word}
                new_counter = counter.copy()
                new_column = list()
                for row in column:
                    if word in row:
                        for token in row:
                            new_counter[token] -= 1
                            if new_counter[token] == 0:
                                del new_counter[token]
                    else:
                        new_column.append(row)
                # add to queue
                queue.append((new_counter, new_column, new_found))
        return result
