from . import Transformation


class OneHot(Transformation):
    """One hot encode a column."""

    def __call__(self, column: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(column)

    def arguments(self, column: pd.Series) -> Tuple[Any]:
        return ()
