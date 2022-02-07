from operator import itemgetter
from numpy import add
from numpy.lib.function_base import select
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from avatar.evaluate import *
from avatar.utilities import prepare
from avatar.select import CHCGASelector
from avatar.filter import StringFilter


def get_titanic():
    titanic = pd.read_csv(Path(__file__).parent / "data/titanic_expanded.csv")
    titanic = prepare(titanic)
    titanic = StringFilter().select(titanic, target="Survived")
    return titanic


def test_ga():
    titanic = get_titanic()

    # make a simple game
    estimator = DecisionTreeClassifier(max_depth=4)
    game = Game(estimator=estimator, judge=DefaultJudge(), rounds=4)

    # inititalise the selector
    selector = CHCGASelector(game, games=500)
    selector.fit(titanic, target="Survived")
    s = selector.select()
    print(s, selector._results[s])

    # print(
    #     selector.select(), max(selector._results.values()), selector._evaluator.played
    # )


if __name__ == "__main__":

    test_ga()
