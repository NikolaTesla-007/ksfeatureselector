"""
Multi-class selection inside a tuned pipeline
=============================================

This example uses ``KSFeatureSelector`` on the multi-class wine dataset and
tunes the number of selected features with ``GridSearchCV``.
"""

# %%
# Set up the pipeline
# -------------------
# For multi-class targets the selector compares classes either pairwise or
# one-against-the-rest, then aggregates the p-values. Here we let the grid
# search choose both the number of features and the comparison strategy.

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ksfeatureselector import KSFeatureSelector

X, y = load_wine(return_X_y=True)

pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("ks", KSFeatureSelector()),
        ("clf", LogisticRegression(max_iter=5000)),
    ]
)

param_grid = {
    "ks__top_n": [3, 5, 8, 13],
    "ks__aggregation_method": ["pairwise", "one-vs-rest"],
}

search = GridSearchCV(pipe, param_grid, cv=5)
search.fit(X, y)

# %%
# Inspect the result
# ------------------
# The grid search reports the best feature count and comparison strategy along
# with the cross-validated accuracy it achieved.

print("Best parameters:", search.best_params_)
print(f"Best CV accuracy: {search.best_score_:.4f}")
