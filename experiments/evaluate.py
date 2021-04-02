"""

Run feature ranking.

"""
from os import name
from avatar.evaluate import RankingEvaluator
import sys
import time
import json
import argparse
import importlib
import itertools
from pathlib import Path
from tqdm import tqdm
from avatar.supervised import *
from avatar.settings import Settings
from avatar.utilities import estimator_to_string


def get_estimator(classifier, classification: bool = True):
    """

    Args:
        classifier: String representation of classifier.
        classification: Whether to get a classifier or a regressor.

    Returns:
        Classifier object.

    """
    name, arguments = classifier.strip(")").split("(")
    # get right task
    if classification:
        name = name.replace("Regressor", "Classifier")
    else:
        name = name.replace("Classifier", "Regressor")
    # try to load the module
    for module in ["sklearn.tree", "sklearn.ensemble", "xgboost"]:
        try:
            m = importlib.import_module(module)
            try:
                clf = getattr(m, name)
                break
            except AttributeError:
                pass
        except ImportError:
            pass
    # parse arguments
    arg = eval("dict({})".format(arguments))
    # hack for XGB
    if "XGB" in name:
        arg["use_label_encoder"] = False
        arg["verbosity"] = 0
    return clf(**arg)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=str)
    parser.add_argument("--estimator", type=str, default=None)
    parser.add_argument("--max", type=int, default=32)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    Settings.verbose = args.verbose

    # load metadata
    file = Path(args.ranking)
    name = file.parent.stem
    with open(file) as metaf:
        meta = json.load(metaf)

    # properties
    classification = meta["type"].lower() == "classification"

    # get estimator
    if args.estimator is None:
        args.estimator = "RandomForestRegressor(max_depth=8)"
    estimator = get_estimator(args.estimator, classification=classification)

    # make result file and check if exists
    result_file = file.parent.parent.parent / "results" / name / file.name
    if result_file.exists() and not args.force:
        sys.exit()

    # then load data
    data = pd.read_csv(meta["data"], index_col=None)

    # play
    evaluator = RankingEvaluator(estimator, max_features=args.max, folds=args.folds)
    evaluator.fit(data, meta["target"], meta["ranking"])

    # collect results
    results = dict(meta)
    results["scores"] = evaluator.scores
    del results["ranking"]

    # write to file
    result_file.parent.mkdir(exist_ok=True, parents=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)