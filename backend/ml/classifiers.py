"""
AQUA / BAYES — Machine Learning Classification and Regression Models
=====================================================================
Modular pipeline for training, calibrating, and saving candidate models:
1. Multinomial Logistic Regression (L2 regularized, balanced class weights)
2. Random Forest Classifier (calibrated, balanced subsample)
3. HistGradientBoosting Classifier (gradient boosted decision trees)
4. Probability calibration via CalibratedClassifierCV (sigmoid / isotonic)
5. Continuous WQI regression models (Ridge, Random Forest Regressor)
"""

from pathlib import Path
import json
import pickle
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / 'backend/ml/artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ['ph', 'conductance', 'dissolved_oxygen', 'temperature']
RISK_CLASSES = ['low', 'moderate', 'high', 'critical']


class CalibratedRiskModel:
    """
    Production-grade container encapsulating standardizer, base estimator,
    and calibrated probability predictor.
    """
    def __init__(self, model_name: str, base_estimator: Any, calibrate: bool = True):
        self.model_name = model_name
        self.base_estimator = base_estimator
        self.calibrate = calibrate
        self.scaler = StandardScaler()
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.classes_: np.ndarray = np.array(RISK_CLASSES)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        X_tr_sc = self.scaler.fit_transform(X_train[FEATURES])
        self.classes_ = np.array(sorted(list(set(y_train))))

        if self.calibrate:
            # CalibratedClassifierCV with 3-fold cross validation for robust, well-calibrated probabilities
            self.calibrated_model = CalibratedClassifierCV(
                estimator=self.base_estimator,
                cv=3
            )
            self.calibrated_model.fit(X_tr_sc, y_train)
        else:
            self.base_estimator.fit(X_tr_sc, y_train)
            self.calibrated_model = None

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self.scaler.transform(X[FEATURES])
        if self.calibrated_model is not None:
            return self.calibrated_model.predict(X_sc)
        return self.base_estimator.predict(X_sc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self.scaler.transform(X[FEATURES])
        if self.calibrated_model is not None:
            probs = self.calibrated_model.predict_proba(X_sc)
            model_classes = self.calibrated_model.classes_
        else:
            probs = self.base_estimator.predict_proba(X_sc)
            model_classes = self.base_estimator.classes_

        # Align probabilities to standard 4 classes: ['low', 'moderate', 'high', 'critical']
        aligned = np.zeros((len(X), len(RISK_CLASSES)))
        for i, cls in enumerate(RISK_CLASSES):
            if cls in model_classes:
                idx = list(model_classes).index(cls)
                aligned[:, i] = probs[:, idx]
        # Normalize in case of numerical precision differences
        sums = aligned.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return aligned / sums

    def predict_record(self, record: Dict[str, float]) -> Dict[str, Any]:
        """
        Fast single-sample prediction for live API requests.
        """
        df = pd.DataFrame([{f: float(record.get(f, 0.5)) for f in FEATURES}])
        probs = self.predict_proba(df)[0]
        prob_dict = {RISK_CLASSES[i]: round(float(probs[i]), 5) for i in range(len(RISK_CLASSES))}
        dominant = max(prob_dict, key=prob_dict.get)
        return {
            'dominant_risk': dominant,
            'probabilities': prob_dict,
            'confidence': round(float(prob_dict[dominant]), 4),
            'model_name': self.model_name
        }

    def save(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> 'CalibratedRiskModel':
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def build_candidate_models() -> Dict[str, CalibratedRiskModel]:
    """
    Factory creating candidate models with regularization and class imbalance handling.
    """
    return {
        'logistic_regression': CalibratedRiskModel(
            'logistic_regression',
            LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
            calibrate=True
        ),
        'random_forest': CalibratedRiskModel(
            'random_forest',
            RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced_subsample', random_state=42),
            calibrate=True
        ),
        'gradient_boosting': CalibratedRiskModel(
            'gradient_boosting',
            HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42),
            calibrate=True
        )
    }


class WQIRegressor:
    """
    Predicts continuous Horton/Brown Water Quality Index (0 to 100).
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_sc = self.scaler.fit_transform(X_train[FEATURES])
        self.model.fit(X_sc, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_sc = self.scaler.transform(X[FEATURES])
        return np.clip(self.model.predict(X_sc), 0.0, 100.0)

    def predict_record(self, record: Dict[str, float]) -> float:
        df = pd.DataFrame([{f: float(record.get(f, 0.5)) for f in FEATURES}])
        return float(self.predict(df)[0])

    def save(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Path) -> 'WQIRegressor':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
