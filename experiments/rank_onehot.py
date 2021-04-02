"""

Run feature ranking.

"""
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
    parser.add_argument("--experiment")
    parser.add_argument("--name", type=str)
    parser.add_argument("--tournament", type=str, default="annealing")
    parser.add_argument("--judge", type=str, default="shap")
    parser.add_argument("--skill", type=str, default="average")
    parser.add_argument("--estimator", type=str, default=None)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--team", default=16)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()

    Settings.verbose = args.verbose

    # load metadata
    file = Path(args.experiment)
    with open(file.parent / "meta.json") as metaf:
        meta = json.load(metaf)

    # properties
    classification = meta["type"].lower() == "classification"

    # get judge
    if "shap" in args.judge:
        judge = SHAPJudge()
    elif "permutation" in args.judge:
        judge = PermutationJudge()
    else:
        judge = DefaultJudge()

    # get estimator
    if args.estimator is None:
        args.estimator = "DecisionTreeRegressor(max_depth=4)"
    estimator = get_estimator(args.estimator, classification=classification)

    # initialise game
    game = Game(
        estimator=estimator, judge=judge, rounds=args.rounds, samples=args.samples
    )

    # initialise pool
    if args.skill == "true":
        pool = TruePool()
    else:
        pool = AveragePool()

    # select tournament
    if "annealing" in args.tournament:
        tournament = AnnealingTournament
    else:
        tournament = Tournament

    if args.team.isdigit():
        args.team = int(args.team)

    # create from parameters
    tournament = tournament(
        game=game,
        pool=pool,
        games=args.games,
        exploration=args.exploration,
        teamsize=args.team,
    )

    # make result file and check if exists
    result_file = (
        file.parent
        / "rankings"
        / args.name
        / "{}.json".format(tournament).replace(" ", "")
    )
    if result_file.exists() and not args.force:
        sys.exit()

    # then load data
    data = pd.read_csv(file, index_col=None)

    # play
    start = time.time()
    tournament.initialise(data, target=meta["target"])
    tournament.play()
    end = time.time()

    # collect results
    results = dict()
    results["ranking"] = tournament.results
    results["parameters"] = tournament.parameters
    results["time"] = end - start
    results["data"] = str(file)
    results["target"] = meta["target"]
    results["type"] = meta["type"]

    # write to file
    result_file.parent.mkdir(exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)