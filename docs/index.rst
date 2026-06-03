ksfeatureselector
=================

A scikit-learn compatible feature selector that ranks features by how well they
separate the classes of a binary or multi-class target using the two-sample
Kolmogorov-Smirnov (K-S) test.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   api

Installation
------------

.. code-block:: bash

   pip install ksfeatureselector

Quick start
-----------

.. code-block:: python

   import numpy as np
   from ksfeatureselector import KSFeatureSelector

   rng = np.random.RandomState(0)
   X = rng.normal(size=(200, 5))
   y = (X[:, 0] + X[:, 1] > 0).astype(int)

   selector = KSFeatureSelector(top_n=2).fit(X, y)
   X_reduced = selector.transform(X)

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
