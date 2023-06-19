"""Evaluate the rankings."""
from time import time
import tqdm
import json
import argparse
from pathlib import Path
from avatar.evaluate import *
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
    data = data.dropna(subset=[meta["target"]])
    return data, meta


def get_estimator(task: str):
    if task == "classification":
        return DecisionTreeClassifier(max_depth=4)
    else:
        return DecisionTreeRegressor(max_depth=4)


def run(experiment_file: Path):

    # load data
    file = Experiment.get_file(experiment_file, "baseline", "selection")
    data, meta = read(experiment_file)

    # get estimator
    estimator = get_estimator(meta["task"])

    # initialise ranker
    ranker = Game(
        estimator=estimator,
        rounds=Experiment.games * Experiment.rounds,
        samples=min(len(data.index), Experiment.samples),
    )
    ranker.initialise(data, meta["target"])

    # rank with everything
    start = time()
    ranking = ranker.play(set(data.columns) - {meta["target"]}).performances
    ranked = sorted(ranking, key=ranking.get, reverse=True)
    end = time()

    # initialise evaluator
    evaluator = Experiment.get_evaluator(meta["task"])
    evaluator.initialise(data, meta["target"])
    result = evaluator.play(ranked[: Experiment.select])

    # collect data
    data = {"ranking": ranking, "result": result.json, "time": end - start}

    # save
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


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

    bar = tqdm.tqdm(total=len(experiments), desc="Experiment", position=0)
    for experiment in experiments:
        bar.set_postfix_str(experiment.parent.name)
        run(experiment)
        bar.update()
