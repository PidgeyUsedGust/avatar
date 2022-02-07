import pandas as pd
from pathlib import Path
from avatar.utilities import *


def test_prepare():
    nba = pd.read_csv(Path(__file__).parent / "data/nba.csv")
    prepared = prepare(nba)
    print(prepared)
    print(prepared.dtypes)


def test_encode_name():
    assert encode_name("[<]") == ""
    assert encode_name("name]") == "name"
    assert encode_name("na < me") == "na  me"


if __name__ == "__main__":
    test_encode_name()
    test_prepare()
