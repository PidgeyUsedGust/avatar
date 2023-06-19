from pyexpat import ExpatError
import time
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from avatar.evaluate import *
from avatar.select import CHCGASelector, ChunkedSFFSelector
from settings import Experiment


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


def get_estimator(task: str):
    if task == "classification":
        return DecisionTreeClassifier(max_depth=4)
    else:
        return DecisionTreeRegressor(max_depth=4)


# def get_selector(
#     configuration: Dict[str, Union[str, int]], classification: bool
# ) -> Selector:
#     judge = SHAPJudge()
#     # make estimator
#     estimator = get_estimator(configuration["estimator"], classification=classification)
#     # initialise game
#     game = Game(
#         estimator=estimator,
#         judge=judge,
#         rounds=configuration["rounds"],
#         samples=configuration["samples"],
#     )
#     # make and return selector
#     return CHCGASelector(game, games=configuration["games"])


def run(experiment_file: Path):

    # prepare output and check if it doesn't exist
    file = Experiment.get_file(experiment_file, "sffs+chunk", "selection_sffs")
    data, meta = read(experiment_file)

    # initialise
    game = Game(
        estimator=get_estimator(meta["task"]),
        judge=SHAPJudge(),
        rounds=Experiment.rounds,
        samples=min(len(data.index), Experiment.samples),
    )
    selector = ChunkedSFFSelector(
        game,
        games=Experiment.games,
        chunks=16,
        add="chunk",
    )

    # play
    start = time.time()
    selector.fit(data, meta["target"])
    end = time.time()

    # evaluate
    selected = selector.select()
    evaluator = Experiment.get_evaluator(meta["task"])
    evaluator.initialise(data, meta["target"])
    result = evaluator.play(selected)

    # collect result
    result = {"result": result.json, "time": end - start, "total": len(data.columns)}

    with open(file, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    # set settings
    Settings.verbose = args.verbose

    # load experiments
    if args.experiment is not None:
        experiments = [Path("data/processed") / args.experiment / "data_3.csv"]
    else:
        experiments = list(Path("data/processed").glob("**/data_3.csv"))

    bar = tqdm(total=len(experiments), desc="Experiment", position=0)
    for experiment in experiments:
        bar.set_postfix_str(experiment.parent.name)
        run(experiment)
        bar.update()
