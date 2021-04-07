from .expand import Expander
from .language import WranglingLanguage


__all__ = ["Expander", "WranglingLanguage"]


# class Avatar:
#     """Main avatar class."""

#     def __init__(
#         self,
#         language: WranglingLanguage = None,
#         pruner: Filter = None,
#         preselector: Filter = None,
#         featureselector: Selector = None,
#         evaluator: DatasetEvaluator = None,
#         n_iterations: int = 3,
#     ):

#         # initialise components
#         self._language = language or WranglingLanguage()
#         self._pruner = pruner or default_pruner
#         self._preselector = preselector or default_filter
#         self._selector = featureselector or SamplingSelector(
#             iterations=1600,
#             evaluator=FeatureEvaluator(n_folds=4, max_depth=4, n_samples=1000),
#         )
#         self._evaluator = evaluator or DatasetEvaluator(max_depth=12)
#         # initialise state
#         self._data = None
#         self._target = None
#         self._selected = None

#     def bend(self, df: pd.DataFrame, target: Label, iterations: int = 0):
#         """Bend dataframe.

#         Args:
#             iterations: Number of iterations. If not provided, stop
#                 if performance doesn't increase.
#             selection: Method of selecting final features to be returned. Can
#                 be one of "best" or "explain_x" with x the cumulative score
#                 of best features to be returned.

#         """

#         # reset properties
#         self._k = len(df.columns)
#         self._ks = set(range(4, self._k * 2 + 1, 2))
#         self._target = target

#         # initialise to right before wrangling step.
#         self._pruned = self._pruner.select(df, target=target)
#         self._preselected = self._preselector.select(self._pruned, target=target)
#         self._selector.fit(self._preselected, target=target)
#         self._scores = self._selector.scores()

#         self._best_k, self._best_score = self.evaluate(self._preselected, self._scores)
#         i = 1
#         while True:

#             # next iteration of wrangling, pruning and selection
#             wrangled = self._language.expand(self._pruned, target=target)
#             pruned = self._pruner.select(wrangled, target=target)
#             preselected = self._preselector.select(pruned, target=target)

#             # get scores
#             self._selector.fit(preselected, target=target)
#             scores = self._selector.scores()

#             # evaluation
#             k, score = self.evaluate(preselected, scores)

#             # check for stopping
#             if score <= self._best_score:
#                 break
#             if iterations and i >= iterations:
#                 break

#             self._best_k = k
#             self._best_score = score
#             self._pruned = pruned
#             self._preselected = preselected
#             self._scores = scores

#             i += 1

#         ranked = np.argsort(self._scores)[::-1]
#         columns = np.array(self._preselected.columns)
#         return self._preselected[columns[ranked][:k]]

#     def evaluate(self, data, scores):
#         """Evaluate."""

#         self._evaluation = dict()

#         scores_nz = np.count_nonzero(scores)
#         scores_ranks = np.argsort(scores)[::-1]
#         columns = np.array(data.columns)

#         for k in tqdm(self._ks, desc="Evaluating", disable=not Settings.verbose):
#             # stop if using features without relevance
#             if k > scores_nz:
#                 break
#             top = columns[scores_ranks][:k]
#             if self._target not in top:
#                 top = np.append(top, self._target)
#             # evaluate
#             self._evaluator.fit(data[top], target=self._target)
#             self._evaluation[k] = self._evaluator.evaluate()

#         k, score = max(self._evaluation.items(), key=lambda x: x[1])

#         if Settings.verbose:
#             print("Best score of {} at {} features.".format(score, k))

#         return k, score

#     def explain(self, how_much=0.9):
#         ranked = np.argsort(self._scores)[::-1]
#         select = np.searchsorted(np.cumsum(self._scores[ranked]), how_much)
#         return self._preselected[self._preselected.columns[ranked[:select]]]

#     @property
#     def language(self) -> WranglingLanguage:
#         return self._language

# # def bend(df: pd.DataFrame, target: Label = None) -> pd.DataFrame:
# #     """Automatically wrangle dataframe.

# #     Args:
# #         df: Dataset to wrangle.
# #         target: Target column.
# #         model: Model for which performance is optimised. If not provided,
# #             MERCS is used.

# #     """
# #     queue = list()


# def bend_custom(
#     df: pd.DataFrame,
#     target: Optional[Label] = None,
#     language: WranglingLanguage = None,
#     pruning: Filter = None,
#     preselection: Filter = None,
#     featureselection: Selector = None,
#     # featureevaluator: FeatureEvaluator = None,
#     evaluation: DatasetEvaluator = None,
#     n_iterations: int = 3,
#     warm_start=True,
# ) -> pd.DataFrame:
#     """Customizable data bending."""

#     n_features = len(df.columns)
#     features_p = list()
#     accuracies = list()
#     wrangled = set()

#     for i in range(n_iterations):

#         print("> Iteration {}".format(i + 1))
#         print("Starting with {} features".format(len(df.columns)))

#         # pruning
#         pruned = pruning.select(df, target=target)
#         print("Pruning; {} left".format(len(pruned.columns)))

#         # preselection
#         select = preselection.select(pruned, target=target)
#         print("Preselection; {} left".format(len(select.columns)))

#         # feature selection
#         featureselection.fit(select, target=target, start=features_p)
#         features = featureselection.select()
#         if target not in features:
#             features.append(target)
#         print("Best features")
#         display(select[features])

#         # evaluate
#         evaluation.fit(select[features], target=target)
#         accuracy = evaluation.evaluate()
#         accuracies.append(accuracy)
#         print("Accuracy; {}%".format(accuracy))

#         if i >= n_iterations - 1:
#             break

#         # remember which columns were already wrangled
#         wrangled_new = set(pruned.columns)
#         df = language.expand(pruned, target=target, exclude=wrangled)
#         wrangled.update(wrangled_new)

#         # if warm starting, remember features used
#         if warm_start:
#             features_p = features


# # def expand(
# #     df: pd.DataFrame,
# #     target: Label = None,
# #     language: WranglingLanguage = None,
# #     pruning: Filter = None
# #     ) -> pd.DataFrame:
# #     """Perform expansion step.

# #     This includes pruning as those columns will
# #     never be used anymore.

# #     """

# #     pass


# def available_models() -> List[str]:
#     """Get available models.

#     Returns:
#         A list of available models to be passed to `bend`.

#     """
#     from importlib import import_module

#     # list of possible classes
#     possible = [
#         ("xgboost", "XGBClassifier", "XGB"),
#         ("lightgbm", "LGBMClassifier", "LGBM"),
#         ("catboost", "CatBoostClassifier", "CB"),
#         ("wekalearn", "RandomForestClassifier", "weka"),
#     ]
#     models = list()
#     for package, classname, short in possible:
#         try:
#             module = import_module(package)
#             class_ = getattr(module, classname)
#         except:
#             pass
#         finally:
#             models.append(short)
#     return models
