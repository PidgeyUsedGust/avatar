"""

Perform data bending without feature selection. This script generates
the dataframes that should to into feature selection at each iterations.

"""
import json
import time
import argparse
from avatar.expand import Expander
from avatar.settings import Settings
from pathlib import Path
from utilities import read_supervised_experiment


if __name__ == "__main__":

    # turn on verbosity
    Settings.verbose = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    #
    exp = Path(args.experiment)
    out = Path(Path(str(exp).replace("raw", "processed")))
    out.mkdir(parents=True, exist_ok=True)

    # load data and drop columns to ignore
    data, meta = read_supervised_experiment(args.experiment)

    expander = Expander()

    # get result without wrangling
    data = expander.prune_transformation.select(data)
    data = expander.prune_full.select(data)

    # write to file
    data.to_csv(out / "data_0.csv", index=False, na_rep="nan")

    # initialise metadata
    meta = dict(target=meta["target"], type=meta["type"], times=list())
    for i in range(args.iterations):

        start = time.time()
        data = expander.expand(data, exclude=[meta["target"]])
        data.to_csv(out / "data_{}.csv".format(i + 1), index=False, na_rep="nan")
        end = time.time()

        # dump metadata after each iteration
        meta["times"].append(end - start)
        with open(out / "meta.json", "w") as metaf:
            json.dump(meta, metaf)