# KSFeatureSelector

[![CI](https://github.com/NikolaTesla-007/ksfeatureselector/actions/workflows/ci.yml/badge.svg)](https://github.com/NikolaTesla-007/ksfeatureselector/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ksfeatureselector.svg)](https://pypi.org/project/ksfeatureselector/)
[![Downloads](https://static.pepy.tech/badge/ksfeatureselector)](https://pepy.tech/project/ksfeatureselector)
[![Python](https://img.shields.io/pypi/pyversions/ksfeatureselector.svg)](https://pypi.org/project/ksfeatureselector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`KSFeatureSelector` is a scikit-learn compatible feature selector that ranks
features by how well they separate the classes of a binary or multi-class
target, using the two-sample Kolmogorov-Smirnov (K-S) test.

It subclasses scikit-learn's `SelectorMixin`, passes `check_estimator`, and
plugs directly into `Pipeline` and `GridSearchCV`.

## Features

- Ranks features by their K-S test p-value (lower p-value = more discriminative).
- Handles binary and multi-class targets (2 to 10 classes).
- Two class-comparison strategies for multi-class targets:
  - `pairwise`: K-S test between every pair of classes.
  - `one-vs-rest`: each class against the rest.
- Three p-value aggregation methods: `fisher` (default), `min`, `max`.
- Select features by a count (`top_n`) or a p-value threshold (`top_p`).
- Full scikit-learn API: `fit`, `transform`, `get_support`,
  `get_feature_names_out`, `inverse_transform`.
- A `select_ks_features` convenience function for quick one-off selection.

## Installation

```bash
pip install ksfeatureselector
```

From source:

```bash
pip install -e ".[test]"
```

## Usage

### Scikit-learn estimator

```python
import numpy as np
from ksfeatureselector import KSFeatureSelector

rng = np.random.RandomState(0)
X = rng.normal(size=(200, 5))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

selector = KSFeatureSelector(top_n=2).fit(X, y)
X_reduced = selector.transform(X)          # shape (200, 2)
print(selector.get_support())              # boolean mask
print(selector.get_feature_p_values())     # [(name, p_value), ...] best first
```

### In a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("ks", KSFeatureSelector(top_p=0.05)),
    ("clf", LogisticRegression()),
])
pipe.fit(X, y)
```

### Convenience function (DataFrame)

```python
from ksfeatureselector import select_ks_features

selected = select_ks_features(
    df, x_cols=["f1", "f2", "f3"], y_var="target",
    top_p=0.01,
    aggregation_method="one-vs-rest",
    p_value_aggregation_method="min",
)
```

## Example

See [`examples/breast_cancer_example.py`](examples/breast_cancer_example.py) for
a runnable end-to-end example on scikit-learn's breast cancer dataset that ranks
features by their K-S p-value and benchmarks selection inside a `Pipeline`.

## Parameters

| Parameter | Values | Description |
| --- | --- | --- |
| `top_n` | int, optional | Keep this many top-ranked features. |
| `top_p` | float in [0, 1], optional | Keep features with aggregated p-value ≤ this. |
| `aggregation_method` | `"pairwise"`, `"one-vs-rest"` | Class comparison strategy. |
| `p_value_aggregation_method` | `"fisher"`, `"min"`, `"max"` | Per-feature p-value aggregation. |

`top_n` and `top_p` are mutually exclusive. If neither is set, all features are
kept (ranked by p-value).

## License

MIT License — see [LICENSE](LICENSE).

## Author

V Subrahmanya Raghu Ram Kishore Parupudi — pvsrrkishore@gmail.com
