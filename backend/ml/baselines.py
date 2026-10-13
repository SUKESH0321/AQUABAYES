"""
AQUA / BAYES — Baseline Models
===============================
Provides minimal reference baselines against which all complex probabilistic
and machine-learning candidate models must be evaluated:
1. MajorityClassClassifier: Always predicts the most frequent class, with calibrated empirical prevalence.
2. UniformPriorClassifier: Assigns equal 1/K probability to all K classes.
3. PersistenceForecaster: For chronological temporal prediction, predicts the state at t for t+k.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class MajorityClassClassifier:
    """
    Predicts the majority class from the training set, outputting constant
    empirical class prevalence probabilities.
    """
    def __init__(self):
        self.classes_: List[str] = []
        self.majority_class_: str = ''
        self.class_priors_: Dict[str, float] = {}

    def fit(self, X: Any, y: pd.Series):
        y_series = pd.Series(y)
        counts = y_series.value_counts(normalize=True)
        self.classes_ = sorted(counts.index.tolist())
        self.majority_class_ = y_series.mode().iloc[0]
        self.class_priors_ = {c: float(counts.get(c, 0.0)) for c in self.classes_}
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.array([self.majority_class_] * n)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        proba_row = [self.class_priors_[c] for c in self.classes_]
        return np.tile(proba_row, (n, 1))


class UniformPriorClassifier:
    """
    Assigns completely uninformative uniform probabilities (1/K) across all classes.
    """
    def __init__(self, classes: List[str] = None):
        self.classes_ = classes or ['low', 'moderate', 'high', 'critical']

    def fit(self, X: Any, y: Any):
        if y is not None:
            self.classes_ = sorted(list(set(y)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.array([self.classes_[0]] * len(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        k = len(self.classes_)
        return np.full((len(X), k), 1.0 / k)


class PersistenceForecaster:
    """
    Persistence baseline for sequential time-series forecasting:
    predicts that current condition persists unchanged into future horizons.
    """
    def predict(self, current_series: pd.Series) -> pd.Series:
        return current_series.copy()
