"""
Common imports and utilities for notebooks in this repository.

Place the following at the top of notebook cells (or `from src.notebook_imports import *`):

from src.notebook_imports import *

This file centralizes imports used across notebooks so NameError / unresolved
reference problems (e.g., `display`, sklearn helpers) are avoided and behavior
is consistent between interactive and script execution.
"""

import logging
import pickle

# Standard library
import sys
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# IPython display for notebooks (fallback to print for script mode)
try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(obj: object) -> None:  # simple fallback for non-notebook runs
        print(obj)


from sklearn.ensemble import RandomForestClassifier

# sklearn helpers commonly used in notebooks
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB

# Common model imports (add others as needed)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# xgboost is optional in some environments
try:
    import xgboost as xgb  # type: ignore
except Exception:
    xgb = None

# Serialization helpers
import joblib

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Convenience: function to recommend imports before unpickling
def ensure_preimports_for_unpickle():
    """Import modules/classes that are required when loading model files.

    Call this in a notebook _before_ calling pickle.load / joblib.load if the
    serialized objects include custom classes defined in this repository.

    Example:
        from src.notebook_imports import ensure_preimports_for_unpickle
        ensure_preimports_for_unpickle()
        model = joblib.load("pipeline.joblib")
    """
    # Import the local transformers so pickle can resolve their class names.
    # If you add other custom classes that are serialized, import them here.
    try:
        # Ensure repository root is on sys.path so `src` is importable when running
        # notebooks from different working directories.
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    except Exception:
        # Best-effort; if import fails, the user should ensure their PYTHONPATH
        # includes the project root or run notebooks from the repository root.
        logger.debug("Could not import local transformer module; ensure src is on sys.path")


__all__ = [
    "np",
    "pd",
    "display",
    "mutual_info_classif",
    "classification_report",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "train_test_split",
    "GridSearchCV",
    "KNeighborsClassifier",
    "GaussianNB",
    "RandomForestClassifier",
    "SVC",
    "xgb",
    "joblib",
    "pickle",
    "logger",
    "ensure_preimports_for_unpickle",
]
