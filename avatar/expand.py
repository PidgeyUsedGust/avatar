import json
import inspect
import pandas as pd
from tqdm import tqdm
from pandas._typing import Label
from pathlib import Path
from hashlib import sha1
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from .settings import Settings
from .language import WranglingLanguage, WranglingTransformation
from .filter import (
    Filter,
    StackedFilter,
    IdenticalFilter,
    MissingFilter,
    ConstantFilter,
    UniqueFilter,
)

UnpackedTransformation = Tuple[str, List[str]]


class Tracker:
    """Keep track of columns.

    Keep both parent to children and reverse
    for fast traversal.

    """

    def __init__(self):
        """Initialise tracker."""
        self.parents = dict()
        self.children = defaultdict(dict)

    def __contains__(self, value: str) -> bool:
        return value in self.parents

    def update(self, data: Dict[str, Tuple[str, WranglingTransformation]]):
        """

        Args:
            data: Mapping of `column` to `(parent, t)` tuples
                such that `t(parent) == column`.

        """
        for c, (p, t) in data.items():
            unpacked = _unpack(t)
            self.children[p][c] = unpacked
            self.parents[c] = (p, unpacked)

    def ancestors(self, column: str) -> List[str]:
        """Get all ancestors in order.

        Returns:
            Ancestors from in order to top.

        """
        ancestors = list()
        while column in self.parents:
            column = self.parents[column][0]
            ancestors.append(column)
        return ancestors

    def descendants(self, column: str) -> List[str]:
        """Get all descendants.

        Returns:
            List of all descendants of a column
            in no particular order.

        """
        result = list()
        queue = [column]
        while len(queue) > 0:
            children = self.children[queue.pop()].keys()
            queue.extend(children)
            result.extend(children)
        return result

    def family(self, column: str) -> List[str]:
        """Get whole family.

        Returns:
            All columns related to this one.

        """
        family = [column] + self.ancestors(column)
        for ancestor in family[:]:
            for descendant in [ancestor] + self.descendants(ancestor):
                if descendant not in family:
                    family.append(descendant)
        return family

    def lca(self, c1: str, c2: str) -> str:
        """Get lowest common ancestor."""
        a1 = self.ancestors(c1)
        a2 = self.ancestors(c2)
        return next((e for e in a1 if e in a2), None)

    def lca_transformations(
        self, c1: str, c2: str
    ) -> Tuple[UnpackedTransformation, UnpackedTransformation]:
        """Get transformations from LCA to columns.

        Returns:
            A tuple of unpacked transformations t1 and t2
            such that c1 was generated through t1 and c2
            through t2 starting from the LCA of c1 and c2.

        """
        a1 = [c1] + self.ancestors(c1)
        a2 = [c2] + self.ancestors(c2)
        which = next(e for e in a1 if e in a2)
        i1 = a1.index(which)
        i2 = a2.index(which)
        if i1 > 0 and i2 > 0:
            p1 = a1[i1 - 1]
            p2 = a2[i2 - 1]
            return (self.children[which][p1], self.children[which][p2])
        return None

    def top(self, column: str) -> str:
        """Get top level column.

        Returns:
            Top level column of column.

        """
        ancestors = self.ancestors(column)
        if len(ancestors) > 0:
            return ancestors[-1]
        else:
            return column

    def to_file(self, file: Path):
        """Save to file."""
        with open(file, "w") as f:
            json.dump(self.parents, f, indent=2)

    @classmethod
    def from_file(cls, file: Path) -> "Tracker":
        """Load tracker from file."""
        with open(file) as f:
            parents = json.load(f)
        tracker = cls()
        tracker.parents = parents
        for c, (p, t) in parents.items():
            tracker.children[p][c] = t
        return tracker


class Expander:
    """Expand dataset, prune and update the tracker.

    Also caches all seen columns.

    """

    def __init__(self, language: Optional[WranglingLanguage] = None):
        self.language = language or WranglingLanguage()
        self.tracker = Tracker()
        self.seen = set()
        self.done = set()
        # pruning within each transformation
        self.prune_transformation = StackedFilter(
            [MissingFilter(0.9), ConstantFilter(0.9)]
        )
        # pruning for whole dataframe
        self.prune_full = IdenticalFilter()

    def expand(self, df: pd.DataFrame, exclude: List[Label] = None) -> pd.DataFrame:
        """Expand a dataframe.

        We expand the dataframe by getting all valid
        transformations, applying them and pruning.

        Pruning happens hierarchically—first within a
        transformation, then within a column and then
        for the whole dataframe.

        """
        if exclude is None:
            exclude = list()
        all_dfs = [df]
        all_map = dict()
        for i, column in tqdm(
            df.iteritems(), total=len(df.columns), disable=not Settings.verbose
        ):
            if i not in exclude and i not in self.done:
                new_dfs = list()
                for transformation in self.language.get_transformations(column):
                    new_df = transformation.execute(df)
                    new_df = self.prune_seen(new_df)
                    new_df = self.prune_transformation.select(new_df)
                    new_dfs.append(new_df)
                    # add to the map
                    for new_column in new_df:
                        all_map[new_column] = (i, transformation)
                # found some new columns
                if len(new_dfs) > 0:
                    column_df = pd.concat(new_dfs, axis=1)
                    column_df = self.prune_full.select(column_df)
                    all_dfs.append(column_df)
            self.done.add(i)
        # combine dataframes and prune
        new_df = pd.concat(all_dfs, axis=1)
        new_df = self.prune_full.select(new_df)
        # only keep those not pruned
        to_remove = set(all_map) - set(new_df.columns)
        for r in to_remove:
            if r in all_map:
                del all_map[r]
        self.tracker.update(all_map)
        return new_df

    def prune_seen(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prune columns that have been seen."""
        to_remove = set()
        for i, column in df.iteritems():
            h = _hash(column)
            if h in self.seen:
                to_remove.add(i)
            else:
                self.seen.add(h)
        return df.drop(labels=to_remove, axis=1)

    def reset(self):
        """Resets the cache."""
        self.seen = set()


def _hash(s: pd.Series) -> int:
    """Compute unique hash of series."""
    return sha1(s.values).hexdigest()


def _unpack(t: WranglingTransformation) -> UnpackedTransformation:
    """Unpack a transformation.

    Returns:
        A tuple of name and arguments.

    """
    transformation = t.transformation
    name = transformation.__class__.__name__
    signature = inspect.signature(transformation.__class__)
    parameter = [str(getattr(transformation, p)) for p in signature.parameters]
    return (name, parameter)
