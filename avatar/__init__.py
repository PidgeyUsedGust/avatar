import pandas as pd
from pandas._typing import Label
from typing import List, Type, Optional
from .language import WranglingTransformation, WranglingProgram, WranglingLanguage
from .selection import Selector
from .filter import Filter
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
    target: Optional[Label] = None,
    language: WranglingLanguage = None,
    pruning: Filter = None,
    preselection: Filter = None,
    featureselection: Selector = None,
    # featureevaluator: FeatureEvaluator = None,
    evaluation: DatasetEvaluator = None,
    n_iterations: int = 3,
    warm_start=True,
) -> pd.DataFrame:
    """Customizable data bending."""

    n_features = len(df.columns)
    features_p = list()
    accuracies = list()
    wrangled = set()

    for i in range(n_iterations):

        print("> Iteration {}".format(i + 1))
        print("Starting with {} features".format(len(df.columns)))

        # pruning
        pruned = pruning.select(df, target=target)
        print("Pruning; {} left".format(len(pruned.columns)))

        # preselection
        select = preselection.select(pruned, target=target)
        print("Preselection; {} left".format(len(select.columns)))

        # feature selection
        featureselection.fit(select, target=target, start=features_p)
        features = featureselection.select()
        if target not in features:
            features.append(target)
        print("Best features")
        display(select[features])

        # evaluate
        evaluation.fit(select[features], target=target)
        accuracy = evaluation.evaluate()
        accuracies.append(accuracy)
        print("Accuracy; {}%".format(accuracy))

        if i >= n_iterations - 1:
            break

        # remember which columns were already wrangled
        wrangled_new = set(pruned.columns)
        df = language.expand(pruned, target=target, exclude=wrangled)
        wrangled.update(wrangled_new)

        # if warm starting, remember features used 
        if warm_start:
            features_p = features


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
