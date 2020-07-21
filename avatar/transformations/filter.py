import re
import string
import itertools
import pandas as pd
from typing import List, Tuple, Any


class Filter:
    """Generic filter.
    
    Attributes:
        keep_min: Proportion of values to keep.

    """

    keep_min = 0.8

    def __call__(self, df: pd.DataFrame, column) -> pd.DataFrame:
        return df[df[column].map(self.keep)]

    def keep(self, value) -> bool:
        return True

    @classmethod
    def arguments(cls, df: pd.Series) -> List[Tuple[Any]]:
        return []


class PatternFilter(Filter):
    """Filter based on regular expression patterns.
    
    This requires the whole string to match the pattern, not
    just part of it.

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

    def __init__(self, pattern: str):
        self._pattern = pattern

    def keep(self, value: str) -> bool:
        return re.search(self._pattern, value) is not None

    @classmethod
    def arguments(cls, column: pd.Series) -> List[Tuple[str]]:
        # translate and generate patterns
        candidates = column.map(cls.translate).map(cls.pattern)
        # find ones that keep sufficient values
        patterns = list()
        for key, value in candidates.value_counts().iteritems():
            if value > (len(column) * cls.keep_min):
                patterns.append(key)
        return patterns

    @classmethod
    def translate(cls, value: str) -> str:
        return value.translate(cls.table)

    @classmethod
    def pattern(cls, value: str) -> str:
        """Patternize a string.
        
        TODO: Generate more interesting patterns.

        """
        pattern = ""
        for k, _ in itertools.groupby(value):
            if k == "0":
                pattern += r"\d+"
            elif k == "a":
                pattern += r"[a-z]+"
            elif k == "A":
                pattern += r"[A-Z]+"
            elif k == " ":
                pattern += r"\w+"
            else:
                pattern += re.escape(k)
        return pattern
