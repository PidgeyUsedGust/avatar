import math
import json
import pandas as pd
from pathlib import Path
from typing import List, Any


def read_supervised_experiment(directory: str):
    path = Path(directory)
    data = pd.read_csv(path / "data.csv")
    with open(path / "meta.json") as metaf:
        meta = json.load(metaf)
    # drop columns to ignore
    if "ignore" in meta:
        data.drop(meta["ignore"], axis=1, inplace=True)
    # drop na
    data = data.dropna(subset=[meta["target"]])
    # set target type
    if meta["type"] == "classification":
        data[meta["target"]] = data[meta["target"]].astype("category")
    else:
        data[meta["target"]] = data[meta["target"]].astype("float")
    return data, meta


def chunk(which: str, of: List[Any]) -> List[Any]:
    a, b = which.split("/")
    a, b = int(a), int(b)
    cz = math.ceil(len(of) / b)
    s = (a - 1) * cz
    e = s + cz
    if a == b:
        e += 1
    return of[s:e]
