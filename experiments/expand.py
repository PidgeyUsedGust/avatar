"""

Perform data bending without feature selection. This script generates
the dataframes that should to into feature selection at each iterations.

"""
import csv
import json
import argparse
from avatar.language import *
from avatar.filter import *
from pathlib import Path
from utilities import read_supervised_experiment


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment")
    parser.add_argument("-i", "--iterations", type=int, default=2)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    args = parser.parse_args()

    # load data
    exp = Path(args.experiment)
    out = Path(Path(str(exp).replace("raw", "processed")))
    out.mkdir(parents=True, exist_ok=True)

    if not args.force and all(
        (out / "data_{}.csv".format(i)).exists() for i in range(args.iterations)
    ):
        print(out.name, "Already exists.")
        quit()

    data, target = read_supervised_experiment(args.experiment)
    language = WranglingLanguage()

    sizes = [len(data.columns)]
    meta = dict(target=target, types=dict())

    for i in range(args.iterations + 1):

        pruned = default_pruner.select(data, target=target)
        presel = default_filter.select(pruned, target=target)

        # # print(out / "data_{}.csv".format(i), len(presel.columns))
        # if 'Split(,)(Split( )(Name)_0)_1' in presel.columns:
        #     print(presel['Split(,)(Split( )(Name)_0)_1'].tolist())

        # dump the preselected
        presel.to_csv(out / "data_{}.csv".format(i), index=False, na_rep="nan")

        # add types to metadata
        meta["types"]["data_{}".format(i)] = presel.dtypes.apply(
            lambda x: x.name
        ).to_dict()

        # add sizes
        sizes.append(len(data.columns))
        sizes.append(len(pruned.columns))
        sizes.append(len(presel.columns))

        # stop
        if i == args.iterations:
            break

        # expand
        data = language.expand(pruned, target=target)

        # dump metadata after each iteration
        meta["chain"] = sizes
        meta["target"] = target
        with open(out / "meta.json", "w") as metaf:
            json.dump(meta, metaf)