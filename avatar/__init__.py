import pandas as pd
from pandas._typing import Label
from typing import List, Type
from .language import WranglingTransformation, WranglingProgram, WranglingLanguage
from .selection import Filter, Selector
from .analysis import FeatureEvaluator, DatasetEvaluator


def bend(df: pd.DataFrame, target: Label = None) -> pd.DataFrame:
    """Automatically wrangle dataframe.
    
    Args:
        df: Dataset to wrangle.
        target: Target column.
        model: Model for which performance is optimised. If not provided,
            MERCS is used.

    """
    queue = list()


def bend_custom(
    df: pd.DataFrame,
    target: Label,
    language: WranglingLanguage,
    pruning: Filter,
    preselecion: Filter,
    featureselection: Selector,
    featureevaluator: FeatureEvaluator,
    evaluator: DatasetEvaluator,
) -> pd.DataFrame:
    """Customizable data bending."""

    pass


def available_models() -> List[str]:
    """Get available models.
    
    Returns:
        A list of available models to be passed to `bend`.

    """
    from importlib import import_module

    # list of possible classes
    possible = [
        ("xgboost", "XGBClassifier", "XGB"),
        ("lightgbm", "LGBMClassifier", "LGBM"),
        ("catboost", "CatBoostClassifier", "CB"),
        ("wekalearn", "RandomForestClassifier", "weka"),
    ]
    models = list()
    for package, classname, short in possible:
        try:
            module = import_module(package)
            class_ = getattr(module, classname)
        except:
            pass
        finally:
            models.append(short)
    return models
