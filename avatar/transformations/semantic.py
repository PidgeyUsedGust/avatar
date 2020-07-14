"""Semantic transformations.

These might require external dependencies and are only
used if the dependencies are found.

"""
from . import Transformation

# fmt: off
try:
    from word2number import w2n

    class WordToNumber(Transformation):
        """Convert words to numbers."""

        def __call__(self, column: pd.Series) -> pd.DataFrame:
            return column.map(w2n.word_to_num).to_frame()

        def arguments(self, column: pd.Series) -> Tuple[Any]:
            return ()

except:
    pass
# fmt: on
