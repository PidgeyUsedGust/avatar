"""Test time it takes to learn models with different sizes."""
from avatar.filter import StringFilter
import time
import random
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from avatar.evaluate import *
from avatar.transformations import Dummies
from avatar.utilities import prepare


def get_titanic():
    return StringFilter().select(
        prepare(
            pd.read_csv(Path(__file__).parent.parent / "test/data/titanic_expanded.csv")
        ),
        "Survived",
    )


def test_time_titanic():
    titanic = get_titanic()
    estimator = DecisionTreeClassifier(max_depth=4)
    game = Game(estimator=estimator, judge=DefaultJudge())
    game.initialise(titanic, "Survived")

    players = list(set(titanic.columns) - {"Survived"})

    start = time.time()
    game.play(players)
    end = time.time()
    print("{} players, {} seconds".format(len(players), end - start))

    start = time.time()
    game.play(random.choices(players, k=4))
    end = time.time()
    print("{} players, {} seconds".format(len(players), end - start))


if __name__ == "__main__":

    test_time_titanic()
