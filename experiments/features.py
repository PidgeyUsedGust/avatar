"""

Run feature selection.

"""
import time
import json
import argparse
import itertools
from pathlib import Path
from tqdm import tqdm
from filelock import FileLock
from avatar.selection import *
from avatar.analysis import *
from utilities import chunk


def generate_selectors():
    """Generate a bunch of selectors."""

    # generate evaluators
    evaluators = list()
    for (folds, depth, sample) in itertools.product([4, 8], [2, 4, 8], [500, 1000]):
        evaluators.append(
            FeatureEvaluator(n_folds=folds, max_depth=depth, n_samples=sample)
        )

    # generate list of selectors
    selectors = list()
    for evaluator in evaluators:
        selectors.append(SamplingSelector(iterations=100, evaluator=evaluator))
        selectors.append(SamplingSelector(iterations=200, evaluator=evaluator))
        selectors.append(SamplingSelector(iterations=400, evaluator=evaluator))
        selectors.append(CHCGASelector(iterations=25, evaluator=evaluator))
        selectors.append(CHCGASelector(iterations=50, evaluator=evaluator))
        selectors.append(CHCGASelector(iterations=100, evaluator=evaluator))

    return selectors


def generate_selectors_debug():

    # generate evaluators
    evaluators = [FeatureEvaluator(folds=4, max_depth=4, n_samples=1000)]

    # generate list of selectors
    selectors = list()
    for evaluator in evaluators:
        selectors.append(SamplingSelector(iterations=100, evaluator=evaluator))
        # selectors.append(CHCGASelector(iterations=25, evaluator=evaluator))
        # selectors.append(SFFSelector(iterations=20, evaluator=evaluator))

    return selectors


def short_name(name):
    name = str(name)
    replace_map = {
        "FeatureEvaluator": "FE",
        "Selector": "",
        "max_depth": "md",
        "evaluator": "e",
        "iterations": "it",
        "sample": "s",
        "population": "pop",
        "folds": "f",
    }
    for s, r in replace_map.items():
        name = name.replace(s, r)
    return name


# def safe_add(values, file):
#     lock = FileLock(str(file) + ".lock")
#     with lock:
#         if file.exists():
#             with open(file) as f:
#                 data = json.load(f)
#         else:
#             data = dict()
#         data.update(values)
#         with open(file, "w") as f:
#             json.dump(data, f)

# def save(values, file):
#     with open(file, "w") as f:
#         json.dump(values, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment")
    parser.add_argument("-c", "--chunk", type=str, default="1/1")
    parser.add_argument("-s", "--selector", type=str, default=None)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    args = parser.parse_args()

    # load data
    exp = Path(args.experiment)

    # read metadata
    with open(exp / "meta.json") as metaf:
        meta = json.load(metaf)

    # read existing features or initialise
    if (exp / "features.json").exists():
        with open(exp / "features.json") as featuref:
            existing = json.load(featuref)
    else:
        existing = dict()
    features = dict()

    # read dataframes and sort by name
    dataframes = [
        pd.read_csv(exp / (file + ".csv"), dtype=dtypes, na_values="nan")
        for file, dtypes in meta["types"].items()
    ]
    dataframes = sorted(dataframes, key=lambda df: len(df.columns))

    # generate and chunk selectors
    if args.selector is None:
        selectors = chunk(args.chunk, generate_selectors())
    else:
        selectors = [eval(args.selector)]

    selector_pbar = tqdm(total=len(selectors), position=0)
    # dataframe_pbar = tqdm(total=len(dataframes), position=1)

    for selector in selectors:
        selector_pbar.set_description(short_name(str(selector)))
        selector_file = Path(exp / "features" / (short_name(selector) + ".json"))
        if not selector_file.exists() or args.force:
            # generate
            features = list()
            for df in dataframes:
                selector.fit(df, target=meta["target"])
                scores = selector.scores()
                features.append(
                    {"scores": scores.tolist(), "columns": df.columns.tolist()}
                )
            # ensure exists
            selector_file.parent.mkdir(exist_ok=True, parents=True)
            with open(selector_file, "w") as f:
                json.dump({str(selector): features}, f)
        selector_pbar.update()

    # safe_add(features, exp / "features.json")
