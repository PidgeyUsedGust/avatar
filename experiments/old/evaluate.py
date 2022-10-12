"""Evaluate the rankings."""
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

    for experiment in tqdm.tqdm(list(experiments), desc="Experiment", position=0):
        for configuration in tqdm.tqdm(
            configurations, desc="Configuration", position=1
        ):
            run(experiment, configuration.stem)
