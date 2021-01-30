"""

Evaluate.


"""
import csv
import json
import argparse
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utilities import read_supervised_experiment
from features import short_name
from avatar.analysis import DatasetEvaluator


def read_experiment(path):

    # read metadata
    with open(path / "meta.json") as f:
        meta = json.load(f)

    # read dataframes
    dataframes = [
        pd.read_csv(path / (file + ".csv"), dtype=dtypes, na_values="nan")
        for file, dtypes in meta["types"].items()
    ]
    dataframes = sorted(dataframes, key=lambda df: len(df.columns))

    # read features
    features = dict()
    for file in (path / "features").glob("*.json"):
        if file.name.startswith("."):
            continue
        with open(file) as f:
            features.update(json.load(f))

    return {"data": dataframes, "target": meta["target"], "features": features}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment")
    parser.add_argument("-d", "--depth", type=int, default=12)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    args = parser.parse_args()

    # load data
    exp = Path(args.experiment)
    out = Path(Path(str(exp).replace("processed", "results"))) / "performance"
    out.mkdir(parents=True, exist_ok=True)

    # read experiment
    experiment = read_experiment(exp)

    evaluator = DatasetEvaluator(max_depth=args.depth)

    feature_pbar = tqdm(total=len(experiment["features"]), position=0)
    iteration_pbar = tqdm(total=0, position=1)
    fold_pbar = tqdm(total=0, position=2)

    for f, results in experiment["features"].items():

        out_file = out / "{}_(depth={}).csv".format(short_name(f), args.depth)
        if out_file.exists() and not args.force:
            feature_pbar.update()
            continue

        runs = list()

        # number in original data
        K = min(len(result["scores"]) for result in results)
        Ks = set(range(K // 2, K * 2 + 1, 2))
        Ks.add(K)

        # to explain
        Es = [0.8, 0.85, 0.9, 0.95]

        iteration_pbar.reset(total=len(results))
        for i, iteration in enumerate(results):
            fold_pbar.reset(total=len(Ks) + len(Es))

            # parse arrays
            scores = np.array(iteration["scores"])
            scores_nz = np.count_nonzero(scores)
            scores_ranks = np.argsort(scores)[::-1]
            columns = np.array(iteration["columns"])

            # explanation based
            for explain in [0.8, 0.85, 0.9, 0.95]:
                # select
                k = np.searchsorted(np.cumsum(scores[scores_ranks]), explain) + 1
                # add to list of possible k
                if k not in Ks:
                    Ks.add(k)
                top = columns[scores_ranks][:k]
                if experiment["target"] not in top:
                    top = np.append(top, experiment["target"])
                # get data
                data = experiment["data"][i][top]
                # evaluate
                evaluator.fit(data, target=experiment["target"])
                runs.append(
                    {
                        "run": f,
                        "k": k,
                        "k_method": "explain {}".format(explain),
                        "i": i,
                        "max depth": args.depth,
                        "accuracy": evaluator.evaluate(),
                    }
                )
                fold_pbar.update()

                # k based
                for k in Ks:
                    # stop if using features without relevance
                    if k > scores_nz:
                        break
                    top = columns[scores_ranks][:k]
                    if experiment["target"] not in top:
                        top = np.append(top, experiment["target"])
                    # get data
                    data = experiment["data"][i][top]
                    # evaluate
                    evaluator.fit(data, target=experiment["target"])
                    runs.append(
                        {
                            "run": f,
                            "k": k,
                            "k_method": "{:.2f} k".format(k / K),
                            "i": i,
                            "max depth": args.depth,
                            "accuracy": evaluator.evaluate(),
                        }
                    )
                    fold_pbar.update()


            iteration_pbar.update()

        df = pd.DataFrame(runs)
        df.to_csv(out_file)

        feature_pbar.update()
