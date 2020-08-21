# TODO

## Feature Selection

* Find functionally dependent columns and use only one of them.
* Find similar columns by learning decision stumps and looking for similar predictions.
  
  * Evaluate speed of doing this (1) by iteratively removing the chosen property or (2) by iteratively feeding only two columns to the feature selector. The first point can work in unsupervised case as well and we can include early stopping, so is probably favoured.

* Perform feature selection on the resulting dataset. We are looking for "good" features to be used in evaluating the overall progress and all "not good" features for wrangling in the next iteration.

  * Use shallow trees over sampled subsets of features and combine feature importances.
  * Use AdaBoost with decission stumps for implicit feature selection,which is nice because we can refer to existing work.

## Transformations

* Fix arguments the `SplitAlign` transformation to limit the number of generated columns as it is too slow.
* Fix arguments to OneHot with threshold on number of unique columns.
  
## Implementation

* Add support for recovering original columns.
* Support renaming a columns to "Target" for implicit target selection. 
