Usage
=====

KSFeatureSelector follows the scikit-learn estimator API: configure it through
the constructor, learn from data with ``fit(X, y)``, and reduce the feature
matrix with ``transform(X)``.

As an estimator
---------------

.. code-block:: python

   from ksfeatureselector import KSFeatureSelector

   selector = KSFeatureSelector(top_n=10).fit(X, y)
   X_reduced = selector.transform(X)
   selector.get_support()            # boolean mask of kept features
   selector.get_feature_p_values()   # (name, p_value) pairs, best first

Selecting by p-value threshold
------------------------------

.. code-block:: python

   selector = KSFeatureSelector(top_p=0.01).fit(X, y)

Multi-class targets
-------------------

For targets with more than two classes, choose a comparison strategy and a
p-value aggregation method:

.. code-block:: python

   selector = KSFeatureSelector(
       top_n=5,
       aggregation_method="one-vs-rest",   # or "pairwise"
       p_value_aggregation_method="fisher",  # or "min", "max"
   ).fit(X, y)

In a pipeline
-------------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.linear_model import LogisticRegression

   pipe = Pipeline([
       ("ks", KSFeatureSelector(top_p=0.05)),
       ("clf", LogisticRegression()),
   ])
   pipe.fit(X, y)

Convenience function for DataFrames
-----------------------------------

.. code-block:: python

   from ksfeatureselector import select_ks_features

   selected = select_ks_features(
       df, x_cols=["f1", "f2", "f3"], y_var="target", top_n=2
   )
