"""Evaluate the rankings."""

"""

Run feature ranking.

"""
import json
import argparse
from pathlib import Path
import tqdm
from xgboost import XGBClassifier, XGBRegressor
from avatar.evaluate import *


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


# def get_estimator(classifier, classification: bool = True):
#     """Turn string into an estimator.

#     Args:
#         classifier: String representation of classifier.
#         classification: Whether to get a classifier or a regressor.

#     Returns:
#         Classifier object.

#     """
#     name, arguments = classifier.strip(")").split("(")
#     # get right task
#     if classification:
#         name = name.replace("Regressor", "Classifier")
#     else:
#         name = name.replace("Classifier", "Regressor")
#     # try to load the module
#     for module in ["sklearn.tree", "sklearn.ensemble", "xgboost"]:
#         try:
#             m = importlib.import_module(module)
#             try:
#                 clf = getattr(m, name)
#                 break
#             except AttributeError:
#                 pass
#         except ImportError:
#             pass
#     # parse arguments
#     arg = eval("dict({})".format(arguments))
#     # hack for XGB
#     if "XGB" in name:
#         arg["use_label_encoder"] = False
#         arg["verbosity"] = 0
#     return clf(**arg)


# def get_tournament(
#     configuration: Dict[str, Union[str, int]], classification: bool
# ) -> Tournament:
#     # parse judge
#     if "shap" in configuration["judge"].lower():
#         judge = SHAPJudge()
#     elif "permutation" in configuration["judge"].lower():
#         judge = PermutationJudge()
#     else:
#         judge = DefaultJudge()
#     # parse skill
#     if "true" in configuration["judge"].lower():
#         pool = TruePool()
#     else:
#         pool = AveragePool()
#     # make into function
#     if isinstance(configuration["team"], str):
#         team = eval(configuration["team"])
#     else:
#         team = configuration["team"]
#     # make estimator
#     estimator = get_estimator(configuration["estimator"], classification=classification)
#     # initialise game
#     game = Game(
#         estimator=estimator,
#         judge=judge,
#         rounds=configuration["rounds"],
#         samples=configuration["samples"],
#     )
#     # make and return tournament
#     return Tournament(
#         game=game,
#         pool=pool,
#         games=configuration["games"],
#         exploration=configuration["exploration"],
#         size=team,
#     )


def get_estimator(task: str):
    if task == "classification":
        return XGBClassifier(use_label_encoder=False, verbosity=0)
    return XGBRegressor(use_label_encoder=False, verbosity=0)


def run(experiment_file: Path, configuration_name: str, force: bool = False):

    # prepare output and check if it doesn't exist
    out_dir = (
        experiment_file.parent.parent.parent
        / "results"
        / experiment_file.parent.name
        / "evaluations"
    )
    out_dir.mkdir(exist_ok=True)
    out_name = "{}.json".format(configuration_name)
    if (out_dir / out_name).exists() and not force:
        return
    else:
        out_dir.mkdir(exist_ok=True)

    # get rankings
    ranking_file = Path(
        "data/results/{}/rankings/{}.json".format(
            experiment_file.parent.name, configuration_name
        )
    )

    # load stuff
    with open(ranking_file) as f:
        ranking = json.load(f)
    data, meta = read(experiment_file)

    # initialise a game
    estimator = get_estimator(meta["task"])
    evaluator = Game(estimator=estimator, rounds=10, samples=len(data.index))
    evaluator.initialise(data, meta["target"])

    # scale ranking
    total = sum(ranking["rankings"].values())
    ranking = {feature: score / total for feature, score in ranking["rankings"].items()}
    ordered = sorted(ranking, key=ranking.get, reverse=True)

    # initialise the result
    results = {"ranking": ranking, "results": list()}

    for i in tqdm.tqdm(range(4, min(36, len(ordered)))):
        team = ordered[:i]
        result = evaluator.play(team)
        results["results"].append({"n": i, "score": result.score})

    with open(out_dir / out_name, "w") as f:
        json.dump(results, f, indent=2)


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

    # load configurations
    configurations = list()
    location = Path(
        args.configuration or "experiments/configurations/grid/default.json"
    )
    if location.is_file():
        configurations.append(location)
    else:
        for file in location.glob("*.json"):
            configurations.append(file)

    for configuration in tqdm.tqdm(configurations, desc="Configuration", position=0):
        run(experiment, configuration.stem)
