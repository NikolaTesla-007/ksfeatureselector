"""
Ranking features on the breast cancer dataset
==============================================

This example ranks the features of scikit-learn's breast cancer dataset by
their Kolmogorov-Smirnov p-value and compares classifier accuracy with and
without K-S based selection inside a pipeline.
"""

# %%
# Load the data and rank features
# -------------------------------
# ``KSFeatureSelector`` assigns each feature an aggregated K-S p-value. A lower
# p-value means the feature separates the two classes more strongly.

from sklearn.datasets import load_breast_cancer

from ksfeatureselector import KSFeatureSelector

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

selector = KSFeatureSelector().fit(X, y)
for name, p in selector.get_feature_p_values()[:10]:
    print(f"{name:30s} p = {p:.3e}")

# %%
# Compare accuracy with and without selection
# -------------------------------------------
# We put the selector in a pipeline and measure cross-validated accuracy using
# only the top ten features, then compare against using all features.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

clf = LogisticRegression(max_iter=5000)

baseline = cross_val_score(clf, X, y, cv=5).mean()

pipe = Pipeline([("ks", KSFeatureSelector(top_n=10)), ("clf", clf)])
selected = cross_val_score(pipe, X, y, cv=5).mean()

print(f"All {X.shape[1]} features : {baseline:.4f} CV accuracy")
print(f"Top 10 K-S features  : {selected:.4f} CV accuracy")

# %%
# Plot the feature ranking
# ------------------------
# The bar chart shows the negative log p-value of each feature, so taller bars
# are more discriminative.

import numpy as np
import matplotlib.pyplot as plt

ranking = selector.get_feature_p_values()
names = [n for n, _ in ranking][:15]
scores = [-np.log10(max(p, 1e-300)) for _, p in ranking][:15]

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(names[::-1], scores[::-1])
ax.set_xlabel("-log10(p-value)")
ax.set_title("Top 15 features by K-S test")
fig.tight_layout()
plt.show()
