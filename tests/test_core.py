import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.estimator_checks import check_estimator

from ksfeatureselector import KSFeatureSelector, select_ks_features, sort_tuple


def make_df(seed=42, n=100):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "feature1": rng.normal(0, 1, n),
            "feature2": rng.normal(1, 1, n),
            "feature3": rng.normal(2, 1, n),
            "target": rng.choice([0, 1], size=n),
        }
    )


# --- sort_tuple -----------------------------------------------------------


def test_sort_tuple():
    tuples = [("a", 0.2), ("b", 0.1), ("c", 0.3)]
    assert sort_tuple(list(tuples)) == [("b", 0.1), ("a", 0.2), ("c", 0.3)]
    assert sort_tuple(list(tuples), reverse=True) == [("c", 0.3), ("a", 0.2), ("b", 0.1)]


def test_invalid_sort_tuple():
    with pytest.raises(AssertionError):
        sort_tuple("not a list")
    with pytest.raises(AssertionError):
        sort_tuple([(1, 0.5)])
    with pytest.raises(AssertionError):
        sort_tuple([("a", -0.1)])


# --- sklearn estimator API ------------------------------------------------


def test_fit_transform_top_n():
    df = make_df()
    X, y = df[["feature1", "feature2", "feature3"]], df["target"]
    selector = KSFeatureSelector(top_n=2).fit(X, y)
    Xt = selector.transform(X)
    assert Xt.shape == (len(df), 2)
    assert selector.get_support().sum() == 2


def test_fit_transform_top_p_one_vs_rest():
    df = make_df()
    X, y = df[["feature1", "feature2", "feature3"]], df["target"]
    selector = KSFeatureSelector(
        top_p=0.9, aggregation_method="one-vs-rest", p_value_aggregation_method="min"
    ).fit(X, y)
    assert selector.transform(X).shape[1] == selector.get_support().sum()


def test_get_support_default_keeps_all():
    df = make_df()
    X, y = df[["feature1", "feature2", "feature3"]], df["target"]
    selector = KSFeatureSelector().fit(X, y)
    assert selector.get_support().all()


def test_feature_names_out():
    df = make_df()
    X, y = df[["feature1", "feature2", "feature3"]], df["target"]
    selector = KSFeatureSelector(top_n=1).fit(X, y)
    names = selector.get_feature_names_out()
    assert len(names) == 1
    assert names[0] in {"feature1", "feature2", "feature3"}


def test_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        KSFeatureSelector(top_n=2).transform(np.zeros((3, 3)))


def test_get_feature_p_values():
    df = make_df()
    selector = KSFeatureSelector().fit(df[["feature1", "feature2"]], df["target"])
    p_vals = selector.get_feature_p_values()
    assert all(isinstance(t, tuple) and len(t) == 2 for t in p_vals)
    # ranking is sorted ascending by p-value
    assert p_vals == sorted(p_vals, key=lambda t: t[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_n": 2, "top_p": 0.1},  # mutually exclusive
        {"top_p": 1.5},  # out of range
        {"aggregation_method": "bogus"},
        {"p_value_aggregation_method": "bogus"},
        {"top_n": 99},  # exceeds n_features
    ],
)
def test_invalid_params(kwargs):
    df = make_df()
    with pytest.raises(ValueError):
        KSFeatureSelector(**kwargs).fit(df[["feature1", "feature2", "feature3"]], df["target"])


def test_constant_target_raises():
    df = make_df()
    df["target"] = "a"
    with pytest.raises(ValueError):
        KSFeatureSelector().fit(df[["feature1"]], df["target"])


def test_works_in_pipeline():
    df = make_df()
    X, y = df[["feature1", "feature2", "feature3"]], df["target"]
    pipe = Pipeline(
        [("ks", KSFeatureSelector(top_n=2)), ("clf", LogisticRegression(max_iter=200))]
    )
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (len(df),)


# --- back-compat wrapper --------------------------------------------------


def test_select_ks_features_wrapper():
    df = make_df()
    features = select_ks_features(
        df, ["feature1", "feature2", "feature3"], "target", top_n=2
    )
    assert isinstance(features, list)
    assert len(features) == 2


def test_select_ks_features_non_numeric_raises():
    df = make_df()
    df["feature1"] = "x"
    with pytest.raises(AssertionError):
        select_ks_features(df, ["feature1"], "target", top_n=1)


# --- sklearn compliance ---------------------------------------------------


def test_check_estimator():
    check_estimator(KSFeatureSelector())
