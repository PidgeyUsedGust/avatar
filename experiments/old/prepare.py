from avatar.utilities import encode_name
import json
import pandas as pd
import numpy as np
from pathlib import Path


if __name__ == "__main__":

    raw = Path("data/raw/")
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    for experiment in raw.glob("*"):
        data = next(experiment.glob("*.csv"))
        print(data)
        # if "food" not in data.parent.name:
        #     continue

        # load metadata
        with open(experiment / "meta.json") as d:
            meta = json.load(d)

        # read experiment
        df = pd.read_csv(data, encoding="utf-8", on_bad_lines="skip")
        df = df.dropna(subset=[meta["target"]])

        # parse target for regression
        if meta["task"] == "regression":
            # apply parser of target
            if "parse" in meta and len(meta["parse"]) > 0:
                df[meta["target"]] = df[meta["target"]].apply(
                    eval("lambda x: {}".format(meta["parse"]))
                )
            # apply filter
            if "filter" in meta and len(meta["filter"]) > 0:
                df = df.query(meta["filter"])
            # perform logscaling
            if meta["scale"]:
                df[meta["target"]] = np.log(df[meta["target"]])

        # check target
        if meta["task"] == "classification":
            s = df[meta["target"]].value_counts()
            df = df[df[meta["target"]].isin(s.index[s > 1])]

        # drop ignore
        df = df.drop(labels=meta["ignore"], axis="columns")
        df.columns = df.columns.map(encode_name)

        processed = out / data.parent.name
        processed.mkdir(exist_ok=True)
        df.to_csv(processed / "data.csv", index=None)
        with open(processed / "meta.json", "w") as f:
            json.dump({"target": meta["target"], "task": meta["task"]}, f, indent=2)
