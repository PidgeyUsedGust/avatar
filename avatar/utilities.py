"""Utility functions."""
import re
import importlib
from typing import Sequence, List, Tuple, Set, Dict
from pandas._typing import Label
from collections import defaultdict
import pandas as pd
import numpy as np


def get_estimator(classifier):
    """

    Args:
        classifier: String representation of classifier.
        classification: Whether to get a classifier or a regressor.

    Returns:
        Estimator object.

    """
    name, arguments = classifier.strip(")").split("(")
    # try to load the module
    for module in ["sklearn.tree", "sklearn.ensemble", "xgboost"]:
        try:
            m = importlib.import_module(module)
            try:
                clf = getattr(m, name)
                break
            except AttributeError:
                pass
        except ImportError:
            pass
    # parse arguments
    arg = eval("dict({})".format(arguments))
    return clf(**arg)


def make_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to lowercase."""
    for column in df:
        if df[column].dtype.name == "object":
            df[column] = df[column].str.lower()
    return df


def encode_name(name: str) -> str:
    """Encode name of feature.

    XGB does not allow [, ] and <.

    Returns:
        Feature name supported by all supported
        estimators.

    """
    return re.sub(r"\[|\]|<", "", name)


def encode_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Encode with labelencoder.

    Returns:
        A new dataframe with columns label encoded and an
        empty dictionary.

    """
    df = df.copy(True)
    for c in df:
        if df[c].dtype.name in ["object", "category"]:
            df[c], _ = pd.factorize(df[c])
    return df.fillna(-1), dict()


def encode_onehot(
    df: pd.DataFrame, threshold: float = 0.1
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One-hot encode data and generate map to original features.

    Args:
        threshold: Filter columns that do not have at least this
            many 1s.

    Returns:
        A one-hot encoded dataframe and mapping of
        columns from df to a list of encoded columns.

    """

    # split everything that is not numerical
    Xo = df.select_dtypes(include="object")
    Xn = df.select_dtypes(exclude="object")

    # encode the object column
    Xd = pd.get_dummies(Xo, prefix_sep="@#$", dtype="bool")
    Xf = Xd.loc[:, Xd.sum() > threshold * len(Xd.index)]

    # create mapping of columns to new columns
    M = defaultdict(list)
    for feature in Xf.columns:
        M[feature.split("@#$")[0]].append(feature)

    # group the columns
    groups = group_columns(Xf)
    for feature, group in groups.items():
        # drop the equal
        Xf = Xf.drop(group, axis=1)
        # replace removed from the group with
        # the one that is kept
        for o in M:
            M[o] = [feature if e in group else e for e in M[o]]

    return pd.concat((Xn, Xf), axis=1).fillna(-1), M


def group_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group identical boolean columns.

    Returns:
        A map from columns in `df` to equal columns
        in df.

    """

    # split
    dB = df.select_dtypes(include="bool")
    dR = df.select_dtypes(exclude="bool")
    dT = dB.T

    # get all duplicated rows
    d = list(dT.index[dT.duplicated(keep=False)])

    # build the map
    M = defaultdict(list)
    while len(d) > 0:
        row = d.pop()
        equals = dT.index[dT.eq(dT.loc[row]).all(axis=1)].tolist()
        equals.remove(row)
        M[row] = equals
        for r in equals:
            if r in d:
                d.remove(r)

    return M


def get_substrings(string: Sequence) -> List[Sequence]:
    """Get substrings of a sequence."""
    substrings = list()
    for i in range(1, len(string) + 1):
        for j in range(0, len(string) - i + 1):
            substring = string[j : j + i]
            if substring not in substrings:
                substrings.append(substring)
    return substrings


def count_unique(df: pd.DataFrame) -> int:
    """Count number of unique values in a dataframe."""
    return pd.unique(df.values.ravel()).size


def normalize(s: pd.Series) -> pd.Series:
    """Min-max scale."""
    mi = s.min()
    ma = s.max()
    if mi == ma:
        if ma == 0:
            return s
        return s / ma
    return (s - s.min()) / (s.max() - s.min())


# def scale_weights(w: List[float]) -> List[float]:
#     """Min-max scale list of floats."""
#     mi = min(w)
#     ma = max(w)
#     if ma == mi:
#         return w
#     return [(v - mi) / (ma - mi) for v in w]


def xor(a: bool, b: bool) -> bool:
    """Logical XOR."""
    return (a and b) or (not a and not b)


def hash_series(s: pd.Series) -> int:
    """Hash the series."""
    pass


def to_mercs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set[Label]]:
    """Encode dataframe for MERCS.

    We assume numerical values will have a numerical dtype and convert
    everything else to nominal integers.

    """
    new = pd.DataFrame()
    nom = set()
    for i, column in enumerate(df):
        if df[column].dtype.name in ["category", "object", "bool"]:
            new[column] = df[column].astype("category").cat.codes.replace(-1, np.nan)
            nom.add(i)
        else:
            new[column] = df[column]
    return new, nom


def to_m_codes(columns: pd.Index, target: Label):
    """Generate m_codes for a target."""
    if target is None:
        return None
    m_codes = np.zeros((1, len(columns)))
    m_codes[0, columns.get_loc(target)] = 1
    return m_codes


def estimator_to_string(estimator):
    """Get string representation of estimator."""
    include = ["max_depth"]
    params = estimator.get_params()
    custom = {p: v for p, v in params.items() if p in include}
    return "{}({})".format(
        estimator.__class__.__name__.replace("Classifier", "").replace("Regressor", ""),
        ",".join("{}={}".format(k, v) for k, v in custom.items()),
    )
