import json
import warnings
import argparse
from tqdm import tqdm
from time import time
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from avatar.evaluate import *
from avatar.ranking import *
from settings import Experiment

warnings.filterwarnings(action="ignore", category=UserWarning)


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
    file = Experiment.get_file(experiment_file, experiment_file.stem, "wrangling+tmp")
    data, meta = read(experiment_file)

    # get estimator
    estimator = get_estimator(meta["task"])

    # initialise ranker
    game = Game(
        estimator=estimator,
        judge=SHAPJudge(),
        rounds=Experiment.rounds,
        samples=min(len(data.index), Experiment.samples),
    )
    tournament = Tournament(
        game=game,
        pool=AveragePool(),
        games=Experiment.games,
        size=16,
        exploration=0,
    )
    tournament.initialise(data, meta["target"])

    # play the tournament
    start = time()
    tournament.play()
    end = time()

    # extract ranking
    print(tournament.pool._counts)
    ranking = tournament.ratings
    ranked = sorted(ranking, key=ranking.get, reverse=True)

    # initialise evaluator
    evaluator = Experiment.get_evaluator(meta["task"])
    evaluator.initialise(data, meta["target"])
    result = evaluator.play(ranked[: Experiment.select])
    result_best = evaluator.play(max(tournament.results, key=lambda r: r.score)._team)

    # collect data
    data = {
        "ranking": ranking,
        "result": result.json,
        "result_best": result_best.json,
        "time": end - start,
    }

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
        experiments = list(
            (Path("data/processed") / args.experiment).glob("data_*.csv")
        )
    else:
        experiments = list(Path("data/processed").glob("**/data_*.csv"))

    bar = tqdm(total=len(experiments), desc="Experiment", position=0)
    for experiment in experiments:
        bar.set_postfix_str(experiment.parent.name)
        run(experiment)
        bar.update()
