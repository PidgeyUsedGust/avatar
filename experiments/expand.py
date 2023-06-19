"""

Perform data bending without feature selection. This script generates
the dataframes that should to into feature selection at each iterations.

"""
import json
import time
import pandas as pd
import argparse
import tqdm
from pathlib import Path
from avatar.expand import Expander
from avatar.utilities import prepare
from avatar.language import WranglingProgram, WranglingTransformation
from avatar.transformations.encoding import Dummies


def read(directory: str):
    path = Path(directory)
    data = pd.read_csv(path / "data.csv")
    with open(path / "meta.json") as f:
        meta = json.load(f)
    # drop columns to ignore
    if "ignore" in meta:
        data = data.drop(meta["ignore"], axis="columns")
    data = data.dropna(subset=[meta["target"]])
    data = prepare(data)
    # set target type
    if meta["task"] == "classification":
        data[meta["target"]] = data[meta["target"]].astype("category")
    else:
        data[meta["target"]] = data[meta["target"]].astype("float")
    return data, meta


def baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Create baseline experiment."""
    program = WranglingProgram()
    for column in df:
        if df[column].dtype == "string":
            program.grow(WranglingTransformation(column, Dummies(), True))
    return program(df)


def run(experiment: Path, iterations: int):
    # prepare output dir
    out = experiment.parent.parent / "results" / experiment.name
    out.mkdir(exist_ok=True, parents=True)
    # load data and drop columns to ignore
    data, meta = read(experiment)
    expander = Expander()
    # generate baseline
    base = baseline(data)
    base = expander.prune.select(base)
    base.to_csv(experiment / "data_0.csv", index=False, na_rep="nan")
    # expand
    times = list()
    for i in tqdm.tqdm(range(iterations)):
        print(data.columns)
        # run expansion
        start = time.time()
        data = expander.expand(data, exclude=[meta["target"]])
        end = time.time()
        # write to file
        data.select_dtypes(exclude="string").to_csv(
            experiment / "data_{}.csv".format(i + 1), index=False, na_rep="nan"
        )
        times.append(end - start)
    # write times to file
    with open(out / "expand.json", "w") as f:
        json.dump(times, f)


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
        run(experiment, args.iterations)
