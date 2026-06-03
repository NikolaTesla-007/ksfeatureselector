# Contributing

Contributions are welcome. This document covers the basics for getting set up.

## Development setup

```bash
git clone https://github.com/NikolaTesla-007/ksfeatureselector.git
cd ksfeatureselector
pip install -e ".[test]"
```

## Running the tests

```bash
pytest -q
```

The suite includes scikit-learn's `check_estimator` conformance test, so any
change to the estimator must keep that passing.

## Guidelines

- Keep the estimator scikit-learn compatible: parameters live on `__init__`,
  state learned in `fit` ends in attributes with a trailing underscore, and
  `transform` follows the `SelectorMixin` contract.
- Add or update tests for any behavior change.
- Follow the existing NumPy-style docstrings.
- CI runs on Python 3.9, 3.11, and 3.12 on Linux and Windows; make sure your
  change passes there.

## Reporting issues

Open an issue at
https://github.com/NikolaTesla-007/ksfeatureselector/issues with a minimal
reproducible example.
