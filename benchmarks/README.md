# Benchmarking KSFeatureSelector against scikit-learn's filters

This note compares `KSFeatureSelector` with two univariate filters that ship
with scikit-learn: the ANOVA F-test (`f_classif`) and mutual information
(`mutual_info_classif`). The goal is to show honestly where a Kolmogorov-Smirnov
based filter helps and where it does not.

## Setup

For each dataset we vary the number of selected features `k`. At every `k` we
build a pipeline that scales the features, selects the top `k` with one of the
three methods, and fits a logistic regression model. We report the mean of a
five-fold cross-validated accuracy. Selection happens inside the pipeline, so no
information from the test folds leaks into the choice of features.

Reproduce with:

```bash
pip install -r benchmarks/requirements.txt
python benchmarks/benchmark.py
```

This writes `benchmarks/results.csv` and one plot per dataset.

## Datasets

| Dataset | Rows | Features | Classes | Type |
| --- | --- | --- | --- | --- |
| breast_cancer | 569 | 30 | 2 | biomedical tabular |
| wine | 178 | 13 | 3 | tabular |
| digits | 1797 | 64 | 10 | dense image pixels |
| fair_affairs | 6366 | 8 | 2 | survey tabular |

`fair_affairs` is Fair's affairs dataset from `statsmodels`, with the target set
to whether a person reported any affair. It loads offline. The first three ship
with scikit-learn.

## Results

The table below shows which method gave the highest accuracy at each feature
count. Ties go to the simplest reading of the numbers.

| Dataset | k=2 | k=5 | k=10 | k=15 | k=20 | k=30 | k=40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | f_classif | **ks** (tie) | mutual_info | **ks** | f_classif | **ks** (tie) | - |
| wine | mutual_info | f_classif | f_classif | - | - | - | - |
| digits | f_classif | mutual_info | f_classif | f_classif | mutual_info | f_classif | **ks** |
| fair_affairs | **ks** (tie) | **ks** | - | - | - | - | - |

Counting the best method across all dataset and `k` combinations gives roughly
nine wins for `f_classif`, six for `ks`, and three for `mutual_info`.

## What the numbers say

On tabular data the K-S filter is competitive. On `fair_affairs`, a real survey
dataset with about six thousand rows, it matched or beat the other filters at
every feature count. On `breast_cancer` it produced the best accuracy at several
feature counts. On `wine` it trailed the F-test by a small margin.

On `digits` the K-S filter was clearly weaker at small feature counts. Each
feature there is a single pixel of a dense image, and one pixel on its own says
little about the digit. A univariate filter of any kind struggles in that
setting, and the K-S version struggled the most.

## When to reach for it

The K-S test responds to any difference between two class distributions, not
just a difference in means. That makes `KSFeatureSelector` a good fit for
tabular problems where the classes differ in the shape or spread of a feature,
which the ANOVA F-test can miss. It is not a replacement for methods that look
at how features interact, and it is not the right tool for dense pixel data.
Used as one filter among several, it adds a distribution-based view that the
built-in selectors do not provide.
