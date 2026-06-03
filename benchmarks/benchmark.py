"""Benchmark KSFeatureSelector against scikit-learn's univariate filters.

Compares K-S based selection with ``f_classif`` and ``mutual_info_classif`` on
several public classification datasets. For each dataset and each number of
selected features ``k``, we report the mean cross-validated accuracy of a
fixed downstream classifier trained on the selected features.

Run with:

    python benchmarks/benchmark.py

Outputs a CSV (benchmarks/results.csv) and, if matplotlib is available, one
PNG per dataset under benchmarks/.
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_wine,
    load_digits,
)
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ksfeatureselector import KSFeatureSelector

warnings.filterwarnings("ignore")

RANDOM_STATE = 0
CV = 5


def _openml_tabular(name, data_id):
    """Fetch an OpenML dataset, keep numeric columns, drop rows with NaNs.

    Returns (name, X, y) or None if the dataset cannot be fetched (e.g. no
    network), so the benchmark still runs offline on the built-in datasets.
    """
    try:
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"skip {name}: could not fetch from OpenML ({type(exc).__name__})")
        return None
    X = d.data.select_dtypes("number")
    keep = X.notna().all(axis=1)
    X = X.loc[keep]
    y = d.target.loc[keep]
    X = X.loc[:, X.std() > 0]
    return (name, X.reset_index(drop=True), y.reset_index(drop=True))


def get_datasets():
    """Return a list of (name, X, y) tuples of public datasets.

    The built-in scikit-learn datasets always run. Additional tabular datasets
    are pulled from OpenML when a network connection is available; they are
    skipped gracefully otherwise so the benchmark stays reproducible offline.
    """
    datasets = []

    bc = load_breast_cancer(as_frame=True)
    datasets.append(("breast_cancer", bc.data, bc.target))

    wine = load_wine(as_frame=True)
    datasets.append(("wine", wine.data, wine.target))

    digits = load_digits(as_frame=True)
    datasets.append(("digits", digits.data.loc[:, digits.data.std() > 0], digits.target))

    # Real tabular dataset bundled with statsmodels (loads offline, no network).
    # "fair" (Fair's affairs data): predict whether a person had an affair from
    # continuous/ordinal survey features. A genuine tabular classification case.
    try:
        import statsmodels.api as sm

        d = sm.datasets.fair.load_pandas().data
        y = (d["affairs"] > 0).astype(int)
        X = d.drop(columns=["affairs"]).select_dtypes("number")
        X = X.loc[:, X.std() > 0]
        datasets.append(("fair_affairs", X.reset_index(drop=True), y.reset_index(drop=True)))
    except Exception as exc:
        print(f"skip fair_affairs: statsmodels unavailable ({type(exc).__name__})")

    # Optional tabular datasets with distributional class differences (KS's
    # natural home). Pulled from OpenML when a network connection is available;
    # skipped gracefully otherwise so the benchmark stays reproducible offline.
    # data_id values are stable OpenML identifiers.
    for name, data_id in [
        ("credit-g", 31),      # German credit (credit risk, binary)
        ("diabetes", 37),      # Pima Indians diabetes (binary)
        ("ionosphere", 59),    # Ionosphere radar returns (binary)
        ("spambase", 44),      # Spam classification (binary, 57 features)
        ("sonar", 40),         # Sonar mines vs rocks (binary)
    ]:
        ds = _openml_tabular(name, data_id)
        if ds is not None and ds[1].shape[1] >= 2:
            datasets.append(ds)

    return datasets


def make_selector(method, k):
    """Build a fitted-on-fit selector transformer for the given method."""
    if method == "ks":
        return KSFeatureSelector(top_n=k)
    if method == "f_classif":
        return SelectKBest(f_classif, k=k)
    if method == "mutual_info":
        return SelectKBest(
            lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=k
        )
    raise ValueError(method)


def evaluate(name, X, y):
    n_features = X.shape[1]
    ks_grid = sorted({k for k in [2, 5, 10, 15, 20, 30, 40] if k <= n_features})
    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)

    rows = []
    for k in ks_grid:
        for method in ["ks", "f_classif", "mutual_info"]:
            pipe = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("select", make_selector(method, k)),
                    ("clf", clf),
                ]
            )
            scores = cross_val_score(pipe, X, y, cv=CV, scoring="accuracy")
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "k": k,
                    "accuracy": scores.mean(),
                    "std": scores.std(),
                }
            )
            print(
                f"{name:14s} {method:12s} k={k:3d}  "
                f"acc={scores.mean():.4f} +/- {scores.std():.4f}"
            )
    return rows


def maybe_plot(df):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    for name, sub in df.groupby("dataset"):
        plt.figure(figsize=(6, 4))
        for method, msub in sub.groupby("method"):
            msub = msub.sort_values("k")
            plt.plot(msub["k"], msub["accuracy"], marker="o", label=method)
        plt.title(f"Feature selection on {name}")
        plt.xlabel("Number of selected features (k)")
        plt.ylabel(f"{CV}-fold CV accuracy")
        plt.legend()
        plt.tight_layout()
        out = f"benchmarks/{name}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"wrote {out}")


def main():
    all_rows = []
    for name, X, y in get_datasets():
        all_rows.extend(evaluate(name, X, y))
    df = pd.DataFrame(all_rows)
    df.to_csv("benchmarks/results.csv", index=False)
    print("\nwrote benchmarks/results.csv")

    # Print a compact win/tie summary: for each (dataset, k), who is best.
    print("\n=== best method per (dataset, k) ===")
    idx = df.groupby(["dataset", "k"])["accuracy"].idxmax()
    best = df.loc[idx, ["dataset", "k", "method", "accuracy"]]
    print(best.to_string(index=False))

    maybe_plot(df)


if __name__ == "__main__":
    main()
