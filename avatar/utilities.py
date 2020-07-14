"""Utility functions."""
from typing import Sequence, List


def get_substrings(string: Sequence) -> List[Sequence]:
    """Get substrings of a sequence."""
    substrings = list()
    for i in range(1, len(string) + 1):
        for j in range(0, len(string) - i + 1):
            substring = string[j : j + i]
            if substring not in substrings:
                substrings.append(substring)
    return substrings