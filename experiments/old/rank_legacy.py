"""

Run feature ranking.

"""
import sys
import time
import json
import argparse
from pathlib import Path
from avatar.supervised import *
from avatar.settings import Settings


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment")
    parser.add_argument("--games", type=int, default=1600)
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

    # load data
    data = pd.read_csv(file, index_col=None)

    # create tournament after data because we need
    # the data to initialise it
    tournament = LegacyTournament(games=args.games)
    tournament.initialise(data, target=meta["target"])

    # make result file and check if exists
    result_file = (
        file.parent
        / "rankings"
        / "legacy"
        / "{}.json".format(tournament).replace(" ", "")
    )
    if result_file.exists() and not args.force:
        sys.exit()

    start = time.time()
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