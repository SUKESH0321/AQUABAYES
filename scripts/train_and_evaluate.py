#!/usr/bin/env python3
"""
AQUA / BAYES — Reproducible Training and Evaluation Pipeline
=============================================================
Trains candidate models on leak-free chronological training folds,
evaluates against baselines on genuinely held-out test sets,
measures probabilistic calibration (Brier score, ECE), and saves model artifacts.
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ml.baselines import MajorityClassClassifier, UniformPriorClassifier
from backend.ml.classifiers import build_candidate_models, WQIRegressor, ARTIFACTS_DIR
from backend.ml.evaluate import evaluate_classifier, evaluate_regressor
from backend.bayes.network import infer, RISK_STATES

PROCESSED_DIR = ROOT / 'backend/data/processed'
FEATURES = ['ph', 'conductance', 'dissolved_oxygen', 'temperature']


def evaluate_bayesian_network(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs exact Bayesian Network inference on df, caching the 81 discrete state combinations
    for ultra-fast evaluation across thousands of samples.
    """
    cache = {}
    preds = []
    probs_list = []
    for _, row in df.iterrows():
        state_key = tuple(str(row.get(f"{f}_state", '')) for f in FEATURES)
        if state_key not in cache:
            ev = {f: float(row[f]) for f in FEATURES}
            res = infer(ev)
            dist = res['posterior']['overall_risk']
            prob_vec = [dist[s] for s in RISK_STATES]
            dom = max(dist, key=dist.get)
            cache[state_key] = (dom, prob_vec)
        dom, prob_vec = cache[state_key]
        preds.append(dom)
        probs_list.append(prob_vec)
    return np.array(preds), np.array(probs_list)


def main():
    print("=" * 70)
    print("AQUA / BAYES: REPRODUCIBLE MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 70)

    train_path = PROCESSED_DIR / 'train_stations.csv'
    val_path = PROCESSED_DIR / 'val_stations.csv'
    test_path = PROCESSED_DIR / 'test_stations.csv'

    if not train_path.exists():
        print("Training dataset not found. Running data pipeline first...")
        from backend.data.pipeline import run_pipeline
        run_pipeline()

    print(f"Loading chronological splits from {PROCESSED_DIR}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Dataset summary: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    print(f"Train class distribution: {train_df['overall_risk'].value_counts().to_dict()}")

    X_train, y_train = train_df[FEATURES], train_df['overall_risk']
    X_val, y_val = val_df[FEATURES], val_df['overall_risk']
    X_test, y_test = test_df[FEATURES], test_df['overall_risk']

    report = {
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'split_type': 'Chronological holdout (no lookahead leakage)',
            'features': FEATURES,
            'classes': RISK_STATES
        },
        'validation_metrics': {},
        'test_metrics': {},
        'model_ranking': []
    }

    models_to_evaluate = {}

    # 1. Baseline: Majority Class
    print("\n[1/6] Training Majority Class Baseline...")
    majority_model = MajorityClassClassifier()
    majority_model.fit(X_train, y_train)
    models_to_evaluate['majority_class_baseline'] = majority_model

    # 2. Baseline: Uniform Prior
    print("[2/6] Initializing Uniform Prior Baseline...")
    uniform_model = UniformPriorClassifier(classes=RISK_STATES)
    uniform_model.fit(X_train, y_train)
    models_to_evaluate['uniform_prior_baseline'] = uniform_model

    # 3. Tabular ML Candidates
    tabular_candidates = build_candidate_models()
    for name, model in tabular_candidates.items():
        print(f"Training & Calibrating {name}...")
        model.fit(X_train, y_train, X_val, y_val)
        models_to_evaluate[name] = model

    # Evaluate all models on Validation split
    print("\n" + "-" * 70)
    print("EVALUATION ON VALIDATION SPLIT:")
    print("-" * 70)
    val_brier_scores = {}

    for name, model in models_to_evaluate.items():
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)
        metrics = evaluate_classifier(y_val, preds, probs, classes=RISK_STATES)
        report['validation_metrics'][name] = metrics
        val_brier_scores[name] = metrics['brier_score']
        print(f"[{name.upper()}] Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Brier: {metrics['brier_score']:.4f} | ECE: {metrics['expected_calibration_error']:.4f}")

    # Evaluate Bayesian Network on Validation
    print("Evaluating Bayesian Network (exact VariableElimination) on Validation...")
    bn_val_preds, bn_val_probs = evaluate_bayesian_network(val_df)
    bn_val_metrics = evaluate_classifier(y_val, bn_val_preds, bn_val_probs, classes=RISK_STATES)
    report['validation_metrics']['bayesian_network'] = bn_val_metrics
    val_brier_scores['bayesian_network'] = bn_val_metrics['brier_score']
    print(f"[BAYESIAN_NETWORK] Acc: {bn_val_metrics['accuracy']:.4f} | Macro F1: {bn_val_metrics['macro_f1']:.4f} | Brier: {bn_val_metrics['brier_score']:.4f} | ECE: {bn_val_metrics['expected_calibration_error']:.4f}")

    # Evaluate on Genuinely Held-Out TEST SPLIT
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON HELD-OUT TEST SPLIT (Zero Leakage):")
    print("=" * 70)

    for name, model in models_to_evaluate.items():
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        metrics = evaluate_classifier(y_test, preds, probs, classes=RISK_STATES)
        report['test_metrics'][name] = metrics
        print(f"[{name.upper()}] Test Acc: {metrics['accuracy']:.4f} | Test F1: {metrics['macro_f1']:.4f} | Brier: {metrics['brier_score']:.4f} | ECE: {metrics['expected_calibration_error']:.4f}")

    # Bayesian Network on Test
    bn_test_preds, bn_test_probs = evaluate_bayesian_network(test_df)
    bn_test_metrics = evaluate_classifier(y_test, bn_test_preds, bn_test_probs, classes=RISK_STATES)
    report['test_metrics']['bayesian_network'] = bn_test_metrics
    print(f"[BAYESIAN_NETWORK] Test Acc: {bn_test_metrics['accuracy']:.4f} | Test F1: {bn_test_metrics['macro_f1']:.4f} | Brier: {bn_test_metrics['brier_score']:.4f} | ECE: {bn_test_metrics['expected_calibration_error']:.4f}")

    # Train Continuous WQI Regressor
    print("\n[Continuous WQI Model] Training Random Forest Regressor...")
    wqi_reg = WQIRegressor()
    wqi_reg.fit(train_df, train_df['wqi'])
    wqi_test_preds = wqi_reg.predict(test_df)
    wqi_metrics = evaluate_regressor(test_df['wqi'], wqi_test_preds)
    report['test_metrics']['wqi_regressor'] = wqi_metrics
    print(f"[WQI Regressor] Test MAE: {wqi_metrics['mae']:.2f} | RMSE: {wqi_metrics['rmse']:.2f} | R²: {wqi_metrics['r2_score']:.4f}")

    # Select Best Performing Model on Validation Brier Score & Macro F1
    # Candidate tabular models ranking
    candidate_ranking = sorted(
        ['random_forest', 'gradient_boosting', 'logistic_regression', 'bayesian_network'],
        key=lambda m: (report['validation_metrics'][m]['brier_score'], -report['validation_metrics'][m]['macro_f1'])
    )
    best_candidate_name = candidate_ranking[0]
    print(f"\nChampion Model Selected by Validation Performance: {best_candidate_name.upper()}")

    # Save artifacts
    best_candidate = models_to_evaluate.get(best_candidate_name)
    if best_candidate is not None:
        best_candidate.save(ARTIFACTS_DIR / 'champion_risk_model.pkl')
        print(f"Saved champion model artifact to {ARTIFACTS_DIR / 'champion_risk_model.pkl'}")

    wqi_reg.save(ARTIFACTS_DIR / 'wqi_regressor.pkl')
    print(f"Saved WQI regressor artifact to {ARTIFACTS_DIR / 'wqi_regressor.pkl'}")

    # Save model card / evaluation summary
    (PROCESSED_DIR / 'evaluation_report.json').write_text(json.dumps(report, indent=2))
    print(f"Full benchmark report saved to {PROCESSED_DIR / 'evaluation_report.json'}")

    return report


if __name__ == '__main__':
    main()
