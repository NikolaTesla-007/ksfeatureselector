"""Kolmogorov-Smirnov based feature selection.

This module provides :class:`KSFeatureSelector`, a scikit-learn compatible
feature selector that ranks features by their ability to separate the classes
of a (binary or multi-class) target using the two-sample Kolmogorov-Smirnov
(K-S) test, together with the :func:`select_ks_features` convenience wrapper.

The estimator follows the scikit-learn API: all configuration is passed to
``__init__``, learning happens in ``fit(X, y)``, and ``transform(X)`` returns
the reduced feature matrix. Because it subclasses
:class:`sklearn.feature_selection.SelectorMixin`, it also exposes
``get_support``, ``inverse_transform`` and ``get_feature_names_out`` and plugs
directly into :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.model_selection.GridSearchCV`.
"""

import warnings
from itertools import combinations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["KSFeatureSelector", "select_ks_features", "sort_tuple"]


def sort_tuple(tup, reverse=False):
    """Sort a list of ``(name, p_value)`` tuples by the second element.

    Parameters
    ----------
    tup : list of tuple
        A list where each element is a ``(str, float)`` tuple. The first
        element is a feature identifier and the second is a p-value in
        ``[0, 1]``.
    reverse : bool, default=False
        If True, sort in descending order of p-value.

    Returns
    -------
    list of tuple
        The input list sorted in place by p-value.

    Raises
    ------
    AssertionError
        If the inputs do not have the expected types or ranges.
    """
    assert isinstance(tup, list), "'tup' must be a list."
    assert isinstance(reverse, bool), "'reverse' must be of type bool."
    for x in tup:
        assert isinstance(x, tuple), "List given has elements of type other than tuple."
        assert len(x) == 2, "Each tuple in 'tup' must have exactly 2 elements."
        assert isinstance(x[0], str), "First element of tuple not a column name (string)."
        assert isinstance(x[1], (int, float)), "Second element of tuple not a number."
        assert 0 <= x[1] <= 1, "P-value out of bounds (must be between 0 and 1)."

    tup.sort(key=lambda x: x[1], reverse=reverse)
    return tup


def _aggregate_p_values(p_values_for_feature, method):
    """Aggregate per-comparison p-values into a single value for one feature.

    Parameters
    ----------
    p_values_for_feature : list of float
        Individual p-values from the pairwise or one-vs-rest comparisons.
    method : {'fisher', 'min', 'max'}
        Aggregation strategy.

    Returns
    -------
    float
        The aggregated p-value in ``[0, 1]``. Returns ``1.0`` when no
        comparison could be performed.
    """
    if not p_values_for_feature:
        return 1.0

    if method == "fisher":
        p = np.asarray(p_values_for_feature, dtype=float)
        # Avoid log(0): clip exact zeros to the smallest positive float.
        p[p == 0] = np.finfo(float).eps
        chi_squared_stat = -2.0 * np.sum(np.log(p))
        dof = 2 * len(p)
        return float(stats.chi2.sf(chi_squared_stat, dof))
    if method == "min":
        return float(np.min(p_values_for_feature))
    if method == "max":
        return float(np.max(p_values_for_feature))
    raise ValueError(f"Unknown p_value_aggregation_method: '{method}'.")


class KSFeatureSelector(SelectorMixin, BaseEstimator):
    """Select features using the Kolmogorov-Smirnov test.

    For each feature the two-sample K-S test is applied between groups defined
    by the target. For multi-class targets the comparisons are combined using
    either a ``pairwise`` strategy (every pair of classes) or ``one-vs-rest``
    (each class against the rest), and the resulting per-comparison p-values are
    aggregated per feature with Fisher's method, the minimum, or the maximum.
    Features are then ranked by their aggregated p-value (lower is more
    discriminative) and selected with either a ``top_n`` count or a ``top_p``
    p-value threshold.

    Parameters
    ----------
    top_n : int, optional
        Number of top-ranked features to keep. Mutually exclusive with
        ``top_p``. Must be a positive integer no greater than the number of
        input features. If both ``top_n`` and ``top_p`` are ``None`` (the
        default), every feature is kept.
    top_p : float, optional
        Keep features whose aggregated p-value is less than or equal to this
        threshold. Must lie in ``[0, 1]``. Mutually exclusive with ``top_n``.
    aggregation_method : {'pairwise', 'one-vs-rest'}, default='pairwise'
        Strategy used to compare classes for multi-class targets.
    p_value_aggregation_method : {'fisher', 'min', 'max'}, default='fisher'
        How the per-comparison p-values are combined into a single value per
        feature.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during ``fit``. Defined only when ``X`` has
        string feature names.
    p_values_ : ndarray of shape (n_features_in_,)
        Aggregated p-value for each input feature, in input column order.
    ranking_ : list of tuple
        ``(feature_name, aggregated_p_value)`` tuples sorted ascending by
        p-value (most discriminative first).
    support_mask_ : ndarray of shape (n_features_in_,)
        Boolean mask of the selected features.

    Examples
    --------
    >>> import numpy as np
    >>> from ksfeatureselector import KSFeatureSelector
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(100, 3))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> selector = KSFeatureSelector(top_n=1).fit(X, y)
    >>> selector.transform(X).shape
    (100, 1)
    """

    _MAX_ALLOWED_CLASSES = 10
    _MIN_OBS_PER_CATEGORY_WARNING = 10

    def __init__(
        self,
        top_n=None,
        top_p=None,
        aggregation_method="pairwise",
        p_value_aggregation_method="fisher",
    ):
        self.top_n = top_n
        self.top_p = top_p
        self.aggregation_method = aggregation_method
        self.p_value_aggregation_method = p_value_aggregation_method

    def _more_tags(self):  # pragma: no cover - legacy tag hook
        return {"requires_y": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        return tags

    def _check_params(self, n_features):
        if self.aggregation_method not in ("pairwise", "one-vs-rest"):
            raise ValueError(
                "aggregation_method must be 'pairwise' or 'one-vs-rest'; "
                f"got {self.aggregation_method!r}."
            )
        if self.p_value_aggregation_method not in ("fisher", "min", "max"):
            raise ValueError(
                "p_value_aggregation_method must be 'fisher', 'min' or 'max'; "
                f"got {self.p_value_aggregation_method!r}."
            )
        if self.top_n is not None and self.top_p is not None:
            raise ValueError("Only one of 'top_n' or 'top_p' may be set, not both.")
        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError(f"'top_p' must lie in [0, 1]; got {self.top_p}.")
        if self.top_n is not None:
            if not isinstance(self.top_n, (int, np.integer)) or isinstance(self.top_n, bool):
                raise ValueError(f"'top_n' must be an integer; got {self.top_n!r}.")
            if not (1 <= self.top_n <= n_features):
                raise ValueError(
                    f"'top_n' must lie in [1, n_features={n_features}]; got {self.top_n}."
                )

    def _pairwise_p_values(self, feature, y, classes):
        p_values = []
        for c1, c2 in combinations(classes, 2):
            a = feature[y == c1]
            b = feature[y == c2]
            if a.size == 0 or b.size == 0:
                p_values.append(1.0)
            else:
                p_values.append(stats.ks_2samp(a, b).pvalue)
        return p_values

    def _one_vs_rest_p_values(self, feature, y, classes):
        p_values = []
        for c in classes:
            a = feature[y == c]
            b = feature[y != c]
            if a.size == 0 or b.size == 0:
                p_values.append(1.0)
            else:
                p_values.append(stats.ks_2samp(a, b).pvalue)
        return p_values

    def fit(self, X, y):
        """Compute the K-S ranking and the selection mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. A :class:`pandas.DataFrame` is accepted; its column
            names are stored in ``feature_names_in_``.
        y : array-like of shape (n_samples,)
            Class labels (binary or multi-class).

        Returns
        -------
        self : object
            The fitted selector.
        """
        X, y = validate_data(self, X, y, ensure_min_features=1, y_numeric=False)
        self._check_params(self.n_features_in_)

        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError(
                f"The target has {n_classes} class; this selector needs the "
                f"target to have 2 to {self._MAX_ALLOWED_CLASSES} classes."
            )
        if n_classes > self._MAX_ALLOWED_CLASSES:
            raise ValueError(
                f"The target has {n_classes} distinct values; this selector "
                f"supports 2 to {self._MAX_ALLOWED_CLASSES} classes."
            )

        counts = np.array([np.sum(y == c) for c in classes])
        for c, n in zip(classes, counts):
            if n < self._MIN_OBS_PER_CATEGORY_WARNING:
                warnings.warn(
                    f"Class {c!r} has only {n} observations, fewer than the "
                    f"recommended minimum of {self._MIN_OBS_PER_CATEGORY_WARNING} "
                    "for reliable K-S test results.",
                    UserWarning,
                    stacklevel=2,
                )

        names = self._feature_names()
        p_values = np.empty(self.n_features_in_, dtype=float)
        for j in range(self.n_features_in_):
            feature = X[:, j]
            if self.aggregation_method == "pairwise":
                pv = self._pairwise_p_values(feature, y, classes)
            else:
                pv = self._one_vs_rest_p_values(feature, y, classes)
            p_values[j] = _aggregate_p_values(pv, self.p_value_aggregation_method)

        self.p_values_ = p_values
        self.ranking_ = sort_tuple(
            [(names[j], float(p_values[j])) for j in range(self.n_features_in_)]
        )
        self.support_mask_ = self._compute_mask(p_values)
        return self

    def _feature_names(self):
        if hasattr(self, "feature_names_in_"):
            return [str(n) for n in self.feature_names_in_]
        return [f"x{j}" for j in range(self.n_features_in_)]

    def _compute_mask(self, p_values):
        n_features = p_values.shape[0]
        mask = np.zeros(n_features, dtype=bool)
        if self.top_p is not None:
            mask[p_values <= self.top_p] = True
        elif self.top_n is not None:
            # Lowest p-values win; stable order keeps ties in column order.
            order = np.argsort(p_values, kind="stable")
            mask[order[: self.top_n]] = True
        else:
            mask[:] = True
        return mask

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_mask_

    def get_feature_p_values(self):
        """Return ``(feature_name, aggregated_p_value)`` tuples, best first.

        Returns
        -------
        list of tuple
            Features and their aggregated p-values, sorted ascending by
            p-value. Requires the selector to be fitted.
        """
        check_is_fitted(self)
        return self.ranking_


def select_ks_features(
    df,
    x_cols,
    y_var,
    top_n=None,
    top_p=None,
    aggregation_method="pairwise",
    p_value_aggregation_method="fisher",
):
    """Select discriminative features from a DataFrame in one call.

    This convenience wrapper preserves the original DataFrame-oriented API: it
    fits a :class:`KSFeatureSelector` on ``df[x_cols]`` against ``df[y_var]``
    and returns the names of the selected features, ranked best first.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data holding the feature columns and the target column.
    x_cols : list of str
        Names of the numeric feature columns to evaluate.
    y_var : str
        Name of the target column.
    top_n : int, optional
        Keep this many top-ranked features. Mutually exclusive with ``top_p``.
    top_p : float, optional
        Keep features with aggregated p-value at most this value. Mutually
        exclusive with ``top_n``.
    aggregation_method : {'pairwise', 'one-vs-rest'}, default='pairwise'
        Class comparison strategy for multi-class targets.
    p_value_aggregation_method : {'fisher', 'min', 'max'}, default='fisher'
        Per-feature p-value aggregation method.

    Returns
    -------
    list of str
        Selected feature names, sorted ascending by aggregated p-value.
    """
    import pandas as pd

    assert isinstance(df, pd.DataFrame), "'df' must be a pandas DataFrame."
    assert isinstance(x_cols, list), "'x_cols' must be a list of column names."
    assert y_var in df.columns, f"Target variable '{y_var}' does not exist in the dataframe."
    for col in x_cols:
        assert col in df.columns, f"Feature variable '{col}' does not exist in the dataframe."
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise AssertionError(
                f"Not all values in '{col}' are numeric. Please ensure all "
                "feature columns are numeric."
            )

    selector = KSFeatureSelector(
        top_n=top_n,
        top_p=top_p,
        aggregation_method=aggregation_method,
        p_value_aggregation_method=p_value_aggregation_method,
    )
    selector.fit(df[x_cols], df[y_var])
    ranking = selector.get_feature_p_values()
    selected = set(np.asarray(x_cols)[selector.get_support()])
    # Preserve ranking order (best first) while honouring the selection mask.
    return [name for name, _ in ranking if name in selected]
