"""
AQUA / BAYES — Model Evaluation and Calibration Metrics Suite
=============================================================
Calculates rigorous, leakage-free metrics for both deterministic and probabilistic predictions:
1. Classification: Accuracy, Balanced Accuracy, Precision, Recall, F1 (macro and weighted).
2. Discrimination: One-vs-Rest ROC-AUC and PR-AUC.
3. Calibration: Multiclass Brier Score and Expected Calibration Error (ECE) across reliability bins.
4. Precautionary Health Risk: False Negative Rate for High and Critical contamination risks.
5. Regression: MAE, RMSE, R² for continuous WQI estimation.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score,
    mean_absolute_error, root_mean_squared_error, r2_score
)

RISK_CLASSES = ['low', 'moderate', 'high', 'critical']


def compute_expected_calibration_error(
    y_true_indices: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Computes multi-class Expected Calibration Error (ECE) and reliability diagram bins:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_indices)
    bins_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if in_bin.sum() > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            diff = abs(acc_in_bin - conf_in_bin)
            ece += (in_bin.sum() / total_samples) * diff

            bins_data.append({
                'bin_index': i,
                'range': [round(bin_lower, 2), round(bin_upper, 2)],
                'samples': int(in_bin.sum()),
                'accuracy': round(acc_in_bin, 4),
                'confidence': round(conf_in_bin, 4),
                'calibration_gap': round(diff, 4)
            })
        else:
            bins_data.append({
                'bin_index': i,
                'range': [round(bin_lower, 2), round(bin_upper, 2)],
                'samples': 0,
                'accuracy': 0.0,
                'confidence': round((bin_lower + bin_upper) / 2.0, 4),
                'calibration_gap': 0.0
            })

    return round(float(ece), 4), bins_data


def compute_multiclass_brier_score(y_true_indices: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    """
    Multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2
    Lower is better (0.0 is perfect probabilistic foresight).
    """
    n = len(y_true_indices)
    one_hot = np.zeros((n, num_classes))
    for i, idx in enumerate(y_true_indices):
        if 0 <= idx < num_classes:
            one_hot[i, idx] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probs: np.ndarray,
    classes: List[str] = RISK_CLASSES
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of multi-state classification and calibrated probabilities.
    """
    y_true_list = list(y_true)
    y_pred_list = list(y_pred)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true_idx = np.array([class_to_idx.get(y, 0) for y in y_true_list])
    y_pred_idx = np.array([class_to_idx.get(y, 0) for y in y_pred_list])

    acc = float(accuracy_score(y_true_list, y_pred_list))
    bal_acc = float(balanced_accuracy_score(y_true_list, y_pred_list))

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true_list, y_pred_list, labels=classes, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true_list, y_pred_list, labels=classes, average='weighted', zero_division=0
    )

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true_list, y_pred_list, labels=classes, average=None, zero_division=0
    )
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            'precision': round(float(per_class_p[i]), 4),
            'recall': round(float(per_class_r[i]), 4),
            'f1_score': round(float(per_class_f1[i]), 4),
            'support': int(per_class_sup[i])
        }

    # Confusion matrix
    cm = confusion_matrix(y_true_list, y_pred_list, labels=classes)
    cm_dict = {
        'labels': classes,
        'matrix': cm.tolist()
    }

    # Probabilistic metrics
    ece, bins_data = compute_expected_calibration_error(y_true_idx, probs, n_bins=10)
    brier = compute_multiclass_brier_score(y_true_idx, probs, len(classes))

    # ROC-AUC (One-vs-Rest) if at least 2 distinct classes present in ground truth
    unique_true = set(y_true_list)
    roc_auc = None
    pr_auc = None
    if len(unique_true) > 1:
        # Create binary indicator matrix for present classes
        present_indices = [class_to_idx[c] for c in classes if c in unique_true]
        if len(present_indices) >= 2:
            try:
                y_true_binary = np.zeros((len(y_true_idx), len(classes)))
                for i, idx in enumerate(y_true_idx):
                    y_true_binary[i, idx] = 1.0
                roc_auc = round(float(roc_auc_score(y_true_binary[:, present_indices], probs[:, present_indices], multi_class='ovr', average='macro')), 4)
                pr_auc = round(float(average_precision_score(y_true_binary[:, present_indices], probs[:, present_indices], average='macro')), 4)
            except Exception:
                pass

    # High / Critical False Negative Precautionary Analysis:
    # Measures what fraction of hazardous cases were misclassified as low or moderate risk!
    severe_mask = y_true.isin(['high', 'critical'])
    severe_count = int(severe_mask.sum())
    if severe_count > 0:
        severe_false_negatives = int((severe_mask & pd.Series(y_pred_list, index=y_true.index).isin(['low', 'moderate'])).sum())
        severe_fn_rate = round(severe_false_negatives / severe_count, 4)
    else:
        severe_false_negatives = 0
        severe_fn_rate = 0.0

    return {
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'macro_precision': round(float(prec_m), 4),
        'macro_recall': round(float(rec_m), 4),
        'macro_f1': round(float(f1_m), 4),
        'weighted_f1': round(float(f1_w), 4),
        'roc_auc_ovr': roc_auc,
        'pr_auc_ovr': pr_auc,
        'brier_score': brier,
        'expected_calibration_error': ece,
        'calibration_bins': bins_data,
        'per_class': per_class,
        'confusion_matrix': cm_dict,
        'high_critical_false_negatives': severe_false_negatives,
        'high_critical_false_negative_rate': severe_fn_rate,
        'total_evaluated': len(y_true)
    }


def evaluate_regressor(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates continuous Water Quality Index (WQI) predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'mae': round(float(mae), 3),
        'rmse': round(float(rmse), 3),
        'r2_score': round(float(r2), 4)
    }
