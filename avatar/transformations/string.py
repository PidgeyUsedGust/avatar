"""String transformations."""
import re
import string
import itertools
import collections
from typing import Tuple, Any, List, Set
import numpy as np
import pandas as pd

from .base import Transformation
from ..utilities import get_substrings, count_unique


class StringTransformation(Transformation):
    """Base string transformation."""

    allowed = ["object", "category"]


class Split(StringTransformation):
    """Split column by delimiter."""

    def __init__(self, delimiter: str):
        self._delimiter = re.escape(delimiter)

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.split(pat=self._delimiter, expand=True)

    def __str__(self) -> str:
        return "Split({})".format(re.sub(r"\\(.)", r"\1", self._delimiter))

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
                for substring in get_substrings(delimiter):
                    # arguments.add(("{}".format(re.escape(substring)),))
                    arguments.add(("{}".format(substring),))
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
        self._delimiter = delimiter.replace("(", "\(").replace(")", "\)")

    # def __call__(self, column: pd.Series) -> pd.DataFrame:
    #     df = pd.DataFrame()
    #     for i, values in column.str.split(pat=self._delimiter).iteritems():
    #         for value in values:
    #             df.loc[i, value] = value
    #     return df.fillna("")

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.get_dummies(sep=self._delimiter)

    def __str__(self) -> str:
        return "SplitAlign({})".format(self._delimiter)

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        """Get possible delimiters.
        
        Similar to regular split, but filter on number of unique values.

        """

        # compute max number of categories
        if cls.max_categories <= 1:
            max_categories = len(column) * max_categories
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
            self._regex = self.default
        else:
            self._regex = pattern

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        expanded = column.str.extract(pat=self._regex, expand=True)
        expanded = expanded.iloc[:, 0].str.replace(",", ".").astype("float").to_frame()
        return expanded
        # return column.str.extract(pat=self._regex, expand=True).astype("float")

    def __str__(self) -> str:
        return "ExtractNumber({})".format(self._regex[8:-8])

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
        # return "(" + pattern + ")"
        return r"(?:^|\D)(" + pattern + r")(?:\D|$)"


class ExtractNumber(StringTransformation):
    """Exctract first number."""

    default = r"([-+]?(?:(?:\d*[,.]\d+)|(?:\d+[.,]?))(?:[Ee][+-]?\d+)?)"
    table = {ord(value): "0" for value in string.digits}

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        expanded = column.str.extract(pat=self.default, expand=True)
        expanded = expanded.iloc[:, 0].str.replace(",", ".").astype("float").to_frame()
        return expanded

    def __str__(self) -> str:
        return "ExtractNumber()"

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[()]]:
        column = column.dropna().map(cls.translate).drop_duplicates()
        numbers = column.str.extract(pat=cls.default, expand=True).iloc[:, 0]
        if numbers.notna().any():
            return [()]
        return []

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
        # return "(" + pattern + ")"
        return r"(?:^|\D)(" + pattern + r")(?:\D|$)"


class ExtractWord(StringTransformation):
    """Extract a word from a string."""

    beam = 1
    stop = 2

    def __init__(self, words: Set[str]):
        self._words = words
        self._regex = "({})".format("|".join(words))

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return column.str.extract(self._regex)

    def __str__(self) -> str:
        return "ExtractWord([{}])".format(", ".join(sorted(self._words)))

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
        # count = collections.Counter(words[0].value_counts().to_dict())
        count = collections.Counter(list(itertools.chain.from_iterable(words)))
        # queue with counter and words
        queue = [(count, words, set())]
        while len(queue) > 0:
            counter, column, found = queue.pop()
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
