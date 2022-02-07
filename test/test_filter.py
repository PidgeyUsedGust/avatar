import pandas as pd
from avatar.filter import *


def test_tests():

    titanic = pd.read_csv("data/raw/demo/titanic.csv")
    pclass1 = titanic["Pclass"] == 1
    # print(np.unique(titanic.Embarked))
    # FreshFilter._test_real_cat(titanic["Fare"], titanic["Pclass"])
    print(FreshFilter._test_bool_cat(titanic["Survived"], pclass1))


def test_identical():
    titanic = pd.read_csv("test/data/titanic_expanded2.csv")
    print(titanic.shape)
    filtered = IdenticalFilter().select(titanic)
    print(filtered.shape)


if __name__ == "__main__":
    # test_tests()
    test_identical()
