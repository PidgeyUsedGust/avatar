import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from avatar.evaluate import *
from avatar.transformations import Dummies
from avatar.utilities import prepare


def get_titanic():
    return prepare(pd.read_csv(Path(__file__).parent / "data/titanic_expanded.csv"))


def test_game():
    titanic = get_titanic()
    print(titanic["Survived"])
    estimator = DecisionTreeClassifier(max_depth=4)

    game = Game(estimator=estimator, judge=DefaultJudge())
    game.initialise(titanic, "Survived")

    players = {"Pclass", "Sex"}
    print(game.play(players))


if __name__ == "__main__":

    test_game()
