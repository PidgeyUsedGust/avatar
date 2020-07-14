from . import Transformation


class Numerical(Transformation):
    """Make a column numerical."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.to_numeric(column, errors="coerce").to_frame()

    def arguments(self, column: pd.Series) -> List[Tuple[]]:
        return ()


class Categorical(Transformation):
    """Make a column nominal (categorical)."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.Categorical(column).to_frame()
    
    def arguments(self, column: pd.Series) -> List[Tuple[]]:
        return ()
