import pandas as pd
from pandas._typing import Label
from .language import WranglingLanguage
from .filter import Filter


class Expander:
    """Takes care of expanding datasets, column tracking and pruning.
    
    Also caches all seen columns.

    """

    def __init__(self, language: WranglingLanguage, prune: Filter):
        self.language = language
        self.prune = prune
        # maps columns to their children. will have to use
        # this direction more often.
        self.tracking = dict()
        # store hashes of seen columns
        self.seen = set()

    def expand(self, df: pd.DataFrame, target: Label) -> pd.DataFrame:
        pass

    def get_relatives(self, column: str, level: int = -1):
        """Get relatives of element.

        Args:
            level: Number of levels to go up before going down.

        """
        pass


def _hash(s: pd.Series) -> int:
    """Compute unique hash of series."""
    pass