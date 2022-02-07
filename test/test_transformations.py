from pathlib import Path
from avatar.transformations.string import *
from avatar.utilities import prepare


def get_nba():
    return prepare(pd.read_csv(Path(__file__).parent / "data/nba.csv"))


def get_titanic():
    return prepare(pd.read_csv(Path(__file__).parent / "data/titanic.csv"))


def get_gsm():
    return prepare(pd.read_csv(Path(__file__).parent / "data/gsm.csv"))


def test_split():
    nba = get_nba()
    date = Split(", ")(nba["BIRTH_DATE"])
    assert date[0].dtype == "string"
    assert date[1].dtype == "int64"


def test_split_arguments():
    nba = get_nba()
    assert len(Split.arguments(nba["BIRTH_DATE"])) == 2
    assert Split.arguments(nba["HEIGHT"]) == [("-",)]
    titanic = get_titanic()
    # high max difference allows splitting on names
    split = Split.max_difference
    Split.max_difference = 20
    assert len(Split.arguments(titanic["Name"])) > 0
    # low max differce doesn't
    Split.max_difference = 5
    assert len(Split.arguments(titanic["Name"])) == 0
    Split.max_difference = split


def test_splitalign():
    # pass
    # nba = get_nba()
    # date = SplitDummies(" ")(nba["BIRTH_DATE"])
    # gsm = get_gsm()
    # print(SplitDummies(" / ")(gsm["network_technology"]))
    pass


def test_splitalign_arguments():
    gsm = get_gsm()
    assert (" / ",) in SplitDummies.arguments(gsm["network_technology"])
    # titanic = get_titanic()
    # print(SplitDummies.arguments(titanic["Name"]))


def test_extractnumberpattern():
    titanic = get_titanic()
    pattern = ExtractNumberPattern("(?:^|\\D)(\\.\\d+)(?:\\D|$)")
    result = pattern(titanic["Ticket"])
    # print(result.dropna())


def test_extractnumberpattern_arguments():
    titanic = get_titanic()
    assert len(ExtractNumberPattern.arguments(titanic["Ticket"])) > 0


def test_extractnumberk():
    titanic = get_titanic()
    ExtractInteger(0)(titanic["Ticket"])
    ExtractInteger(1)(titanic["Ticket"])


def test_extractnumberk_arguments():
    titanic = get_titanic()
    assert len(ExtractInteger.arguments(titanic["Ticket"])) == 2
    assert len(ExtractInteger.arguments(titanic["Cabin"])) == 4


def test_extractboolean():
    titanic = get_titanic()
    # print(ExtractBoolean("mr")(titanic["Name"]))
    # print(ExtractBoolean("mrs")(titanic["Name"]))


def test_extractboolean_arguments():
    titanic = get_titanic()
    arguments = ExtractBoolean.arguments(titanic["Name"])
    print(arguments)


if __name__ == "__main__":

    test_split()
    test_split_arguments()

    test_splitalign()
    test_splitalign_arguments()

    test_extractnumberpattern()
    test_extractnumberpattern_arguments()

    test_extractnumberk()
    test_extractnumberk_arguments()

    test_extractboolean()
    test_extractboolean_arguments()
