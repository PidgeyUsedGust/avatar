"""String transformations."""
import re
import string
import itertools
import collections
from typing import Tuple, Any, List, Set
import numpy as np
import pandas as pd

from .base import Transformation
from ..utilities import count_unique


class StringTransformation(Transformation):
    """Base string transformation."""

    allowed = ["object", "category"]


class Split(StringTransformation):
    """Split column by delimiter."""

    def __init__(self, delimiter: str):
        self.delimiter = re.escape(delimiter)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.split(pat=self.delimiter, expand=True)

    def __str__(self) -> str:
        return "Split({})".format(re.sub(r"\\(.)", r"\1", self.delimiter))

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Get possible delimiters.

        Get all substrings of non-alphanumeric characters.

        """
        # figure out all non alphanumeric strings and deduplicate them.
        split = column.str.split(pat=r"[a-zA-Z0-9]+")
        split = split[split.astype(str).drop_duplicates().index].dropna()
        # generate delimiters by taking consecutive
        arguments = set()
        for delimiters in split:
            delimiters = [d for d in delimiters if d]
            for delimiter in delimiters:
                arguments.add((delimiter,))
        return arguments


class SplitAlign(StringTransformation):
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

    max_categories = 30

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
        # of categories.
        arguments_all = Split.arguments(column)
        arguments = list()
        for argument in arguments_all:
            n_categories = count_unique(
                column.str.split(pat=re.escape(argument[0]), expand=True)
            )
            if n_categories < max_categories:
                arguments.append(argument)

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
        expanded = column.str.extract(pat=self.pattern, expand=True)
        expanded = expanded.iloc[:, 0].str.replace(",", ".").astype("float").to_frame()
        return expanded

    def __str__(self) -> str:
        return "ExtractNumberPattern({})".format(self.pattern[8:-8])

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        if column.dtype != object:
            return []
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


class ExtractNumberK(StringTransformation):
    """Extract k'th simple number."""

    def __init__(self, k: int = 0):
        self.k = k

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        expanded = column.str.extractall(pat=r"(\d+)").unstack().droplevel(0, axis=1)
        expanded = expanded.iloc[:, self.k].astype("float").to_frame()
        return expanded

    def __str__(self) -> str:
        return "ExtractNumberK({})".format(self.k)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[int]]:
        if column.dtype != object:
            return []
        # get reduced representation
        column = column.dropna().drop_duplicates()
        # get all numbers
        numbers = column.str.extractall(r"(\d+)").unstack().droplevel(0, axis=1)
        # turn into k
        return [(k,) for k in range(len(numbers.columns))]


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
        count = collections.Counter(list(itertools.chain.from_iterable(words)))
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


class ExtractBoolean(StringTransformation):
    """Check whether a single word occurs."""

    occurences = 0.05

    def __init__(self, word: str):
        self.word = word
        self._regex = "[^a-zA-Z]{}[^a-zA-Z]".format(word)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.contains(self._regex, regex=True).astype(bool).to_frame()

    def __str__(self) -> str:
        return "ExtractBoolean({})".format(self.word)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[Set[str]]]:
        """Extract words that occur in at least some rows."""
        n = len(column) * cls.occurences
        words = column.str.extractall(pat=r"([a-zA-Z]+)")
        words = set(words[0].values)
        arguments = set()
        for word in words:
            if (
                column.str.contains(
                    "[^a-zA-Z]{}[^a-zA-Z]".format(word), regex=True
                ).sum()
                >= n
            ):
                arguments.add((word,))
        return arguments


class Lowercase(StringTransformation):
    """Convert all strings to lowercase."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.lower().to_frame()

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        return [()]


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
