import tqdm
import json
import argparse
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from avatar.evaluate import Game


def read(directory: Path):
    data = pd.read_csv(directory / "data_0.csv")
    with open(directory / "meta.json") as f:
        meta = json.load(f)
    # set target type
    if meta["task"] == "classification":
        data[meta["target"]] = data[meta["target"]].astype("category")
    else:
        data[meta["target"]] = data[meta["target"]].astype("float")
    return data, meta


def run(experiment: Path):
    data, meta = read(experiment)
    # make estimator
    if meta["task"] == "classification":
        estimator = XGBClassifier(use_label_encoder=False, verbosity=0)
    else:
        estimator = XGBRegressor(verbosity=0)
    # make game
    game = Game(estimator=estimator, rounds=10, samples=len(data))
    game.initialise(data, target=meta["target"])
    # make team
    team = data.columns.tolist()
    team.remove(meta["target"])
    # play it
    result = game.play(team)
    # write result
    out = experiment.parent.parent / "results" / experiment.name
    with open(out / "baseline.json", "w") as f:
        json.dump(result.json, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()

    if args.experiment is not None:
        experiments = [Path("data/processed") / args.experiment]
    else:
        experiments = [p for p in Path("data/processed").glob("*")]

    for experiment in tqdm.tqdm(experiments):
        run(experiment)
