"""Sphinx configuration for ksfeatureselector documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "ksfeatureselector"
copyright = "2026, V Subrahmanya Raghu Ram Kishore Parupudi"
author = "V Subrahmanya Raghu Ram Kishore Parupudi"

try:
    from ksfeatureselector import __version__ as release
except Exception:  # pragma: no cover
    release = "0.3.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"
autoclass_content = "both"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"ksfeatureselector {version}"
html_static_path = ["_static"]
