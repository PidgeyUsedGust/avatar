"""

Run feature ranking.

"""
from avatar.select import CHCGASelector, ChunkedSFFSelector, Selector
import sys
import time
import json
import argparse
import importlib
from pathlib import Path

import tqdm
from avatar.evaluate import *
from avatar.ranking import AveragePool, TruePool, Tournament


def read(file: Path):
    data = pd.read_csv(file)
    with open(file.parent / "meta.json") as f:
        meta = json.load(f)
    # set target type
    if meta["task"] == "classification":
        data[meta["target"]] = data[meta["target"]].astype("category")
    else:
        data[meta["target"]] = data[meta["target"]].astype("float")
    return data, meta


def get_estimator(classifier, classification: bool = True):
    """Turn string into an estimator.

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


def get_selector(
    configuration: Dict[str, Union[str, int]], classification: bool
) -> Selector:
    # parse judge
    if "shap" in configuration["judge"].lower():
        judge = SHAPJudge()
    elif "permutation" in configuration["judge"].lower():
        judge = PermutationJudge()
    else:
        judge = DefaultJudge()
    # make estimator
    estimator = get_estimator(configuration["estimator"], classification=classification)
    # initialise game
    game = Game(
        estimator=estimator,
        judge=judge,
        rounds=configuration["rounds"],
        samples=configuration["samples"],
    )
    # make and return selector
    return CHCGASelector(game, games=configuration["games"])


def run(experiment_file: Path, configuration_file: Path, force: bool = False):

    # prepare output and check if it doesn't exist
    out_dir = (
        experiment_file.parent.parent.parent
        / "results"
        / experiment_file.parent.name
        / "selections"
        / "ga"
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    out_name = "{}.json".format(configuration_file.stem)
    if (out_dir / out_name).exists() and not force:
        return
    else:
        out_dir.mkdir(exist_ok=True)

    with open(configuration_file) as f:
        configuration = json.load(f)
    data, meta = read(experiment_file)

    # initialise
    selector = get_selector(configuration, meta["task"] == "classification")

    # play
    start = time.time()
    selector.fit(data, meta["target"])
    end = time.time()

    # collect result
    result = {
        "selected": list(selector.select()),
        "time": end - start,
        "parameters": configuration,
        "results": [(list(f), s) for f, s in selector._results.items()],
    }

    with open(out_dir / out_name, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--iteration", default=1)
    parser.add_argument("--configuration", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # set settings
    Settings.verbose = args.verbose

    # load experiments
    if args.experiment is not None:
        experiments = [
            Path("data/processed")
            / args.experiment
            / "data_{}.csv".format(args.iteration)
        ]
    else:
        experiments = Path("data/processed").glob(
            "**/data_{}.csv".format(args.iteration)
        )

    # load configuration files
    configurations = list()
    location = Path(args.configuration or "experiments/configurations/ga/default.json")
    if location.is_file():
        configurations.append(location)
    else:
        for file in location.glob("*.json"):
            configurations.append(file)

    for experiment in tqdm.tqdm(list(experiments), desc="Experiment", position=0):
        for configuration in tqdm.tqdm(
            configurations, desc="Configuration", position=1
        ):
            run(experiment, configuration)
