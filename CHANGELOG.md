# Changelog

## 0.3.0

### Changed (breaking)

- `KSFeatureSelector` is now a full scikit-learn estimator. It subclasses
  `BaseEstimator` and `SelectorMixin`, passes `check_estimator`, and works in
  `Pipeline` and `GridSearchCV`.
- Selection parameters (`top_n`, `top_p`, `aggregation_method`,
  `p_value_aggregation_method`) are now set in the constructor.
- `fit(X, y)` now takes a feature matrix `X` and target `y` (array-like or
  pandas DataFrame/Series) instead of `fit(df, x_cols, y_var)`.
- `transform(X)` now returns the reduced feature matrix, following the
  scikit-learn contract. The previous `transform(top_n=..., top_p=...)` call
  style (which returned a list of column names) has been removed.

### Added

- `get_support`, `get_feature_names_out`, and `inverse_transform` via
  `SelectorMixin`.
- `p_values_`, `ranking_`, and `support_mask_` fitted attributes.
- CI across Python 3.9/3.11/3.12 on Linux and Windows.

### Preserved

- The `select_ks_features(df, x_cols, y_var, ...)` convenience function keeps
  its original DataFrame-oriented signature and return value (a ranked list of
  feature names), so existing one-call usage continues to work.
- The `sort_tuple` helper remains available.

## 0.2.0

- Added multi-class support with `pairwise` / `one-vs-rest` comparison and
  `fisher` / `min` / `max` p-value aggregation.
