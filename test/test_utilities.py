from avatar.utilities import *


def test_substrings():

    print(get_substrings("-"))
    print(get_substrings(".-'"))


def test_encode_name():
    assert encode_name("[<]") == ""
    assert encode_name("name]") == "name"
    assert encode_name("na < me") == "na  me"


if __name__ == "__main__":

    test_substrings()
    test_encode_name()