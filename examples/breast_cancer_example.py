"""Feature selection on a real dataset with KSFeatureSelector.

This example ranks the features of scikit-learn's breast cancer dataset by
their Kolmogorov-Smirnov p-value, then compares classifier accuracy with and
without K-S based selection inside a Pipeline.

Run with:

    python examples/breast_cancer_example.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ksfeatureselector import KSFeatureSelector


def main():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target

    # Rank every feature by its aggregated K-S p-value (lower = more useful).
    selector = KSFeatureSelector().fit(X, y)
    print("Top 10 features by K-S p-value:")
    for name, p in selector.get_feature_p_values()[:10]:
        print(f"  {name:30s} p = {p:.3e}")

    clf = LogisticRegression(max_iter=5000)

    baseline = cross_val_score(clf, X, y, cv=5).mean()

    pipe = Pipeline(
        [("ks", KSFeatureSelector(top_n=10)), ("clf", clf)]
    )
    selected = cross_val_score(pipe, X, y, cv=5).mean()

    print(f"\nAll {X.shape[1]} features : {baseline:.4f} CV accuracy")
    print(f"Top 10 K-S features  : {selected:.4f} CV accuracy")


if __name__ == "__main__":
    main()
