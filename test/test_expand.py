from math import exp
import pandas as pd
from pathlib import Path
from avatar.expand import Expander
from avatar.settings import Settings
from avatar.utilities import prepare


def get_titanic():
    return prepare(pd.read_csv(Path(__file__).parent / "data/titanic.csv"))


def expand_titanic():
    titanic = get_titanic()
    expander = Expander()
    expanded = expander.expand(titanic)
    expanded.to_csv(Path(__file__).parent / "data/titanic_expanded.csv", index=None)
    expanded2 = expander.expand(expanded)
    expanded2.to_csv(Path(__file__).parent / "data/titanic_expanded2.csv", index=None)


if __name__ == "__main__":
    Settings.verbose = True
    expand_titanic()
