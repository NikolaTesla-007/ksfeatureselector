KSFeatureSelector
=================

``KSFeatureSelector`` is a scikit-learn compatible feature selector that ranks
features by how well they separate the classes of a binary or multi-class
target, using the two-sample Kolmogorov-Smirnov (K-S) test. It subclasses
scikit-learn's ``SelectorMixin``, passes ``check_estimator``, and plugs
directly into ``Pipeline`` and ``GridSearchCV``.

Features
--------

- Ranks features by their K-S test p-value (lower p-value is more discriminative).
- Handles binary and multi-class targets (2 to 10 classes).
- Two class-comparison strategies for multi-class targets:

  - ``pairwise``: K-S test between every pair of classes.
  - ``one-vs-rest``: each class against the rest.

- Three p-value aggregation methods: ``fisher`` (default), ``min``, ``max``.
- Select features by a count (``top_n``) or a p-value threshold (``top_p``).
- Full scikit-learn API: ``fit``, ``transform``, ``get_support``,
  ``get_feature_names_out``, ``inverse_transform``.
- A ``select_ks_features`` convenience function for quick one-off selection.

Installation
------------

.. code-block:: bash

   pip install ksfeatureselector

Usage
-----

.. code-block:: python

   import numpy as np
   from ksfeatureselector import KSFeatureSelector

   rng = np.random.RandomState(0)
   X = rng.normal(size=(200, 5))
   y = (X[:, 0] + X[:, 1] > 0).astype(int)

   selector = KSFeatureSelector(top_n=2).fit(X, y)
   X_reduced = selector.transform(X)
   print(selector.get_support())
   print(selector.get_feature_p_values())

In a scikit-learn pipeline:

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression

   pipe = Pipeline([
       ("ks", KSFeatureSelector(top_p=0.05)),
       ("clf", LogisticRegression()),
   ])
   pipe.fit(X, y)

Convenience function for DataFrames:

.. code-block:: python

   from ksfeatureselector import select_ks_features

   selected = select_ks_features(
       df, x_cols=["f1", "f2", "f3"], y_var="target",
       top_p=0.01,
       aggregation_method="one-vs-rest",
       p_value_aggregation_method="min",
   )

Parameters
----------

- **top_n** (``int``, optional): keep this many top-ranked features.
- **top_p** (``float`` in ``[0, 1]``, optional): keep features whose aggregated
  p-value is at most this value.
- **aggregation_method** (``{"pairwise", "one-vs-rest"}``): class comparison
  strategy for multi-class targets.
- **p_value_aggregation_method** (``{"fisher", "min", "max"}``): per-feature
  p-value aggregation method.

``top_n`` and ``top_p`` are mutually exclusive. If neither is set, all features
are kept (ranked by p-value).

License
-------

MIT License.

Author
------

V Subrahmanya Raghu Ram Kishore Parupudi
Email: pvsrrkishore@gmail.com
