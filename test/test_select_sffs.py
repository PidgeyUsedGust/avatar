from operator import itemgetter
from numpy import add
from numpy.lib.function_base import select
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from avatar.evaluate import *
from avatar.utilities import prepare
from avatar.select import ChunkedSFFSelector
from avatar.filter import StringFilter


def get_titanic():
    titanic = pd.read_csv(Path(__file__).parent / "data/titanic_expanded.csv")
    titanic = prepare(titanic)
    titanic = StringFilter().select(titanic, target="Survived")
    return titanic


def test_sffs_chunk():
    titanic = get_titanic()

    # make a simple game
    estimator = DecisionTreeClassifier(max_depth=4)
    game = Game(estimator=estimator, judge=DefaultJudge(), rounds=4)

    # inititalise the selector
    selector = ChunkedSFFSelector(game, games=500, chunks=8, add="best")
    selector.fit(titanic, target="Survived")

    print(
        selector.select(), max(selector._results.values()), selector._evaluator.played
    )


if __name__ == "__main__":

    test_sffs_chunk()
