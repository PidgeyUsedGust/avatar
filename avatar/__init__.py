import pandas as pd
from .language import WranglingTransformation, WranglingProgram


def bend(df: pd.DataFrame, target: int = -1, model=None) -> pd.DataFrame:
    """Automatically wrangle dataframe.
    
    Args:
        df: Dataset to wrangle.
        target: Target column.
        model: Model for which performance is optimised. If not provided,
            MERCS is used.

    """

    queue = list()
