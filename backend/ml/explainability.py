"""
AQUA / BAYES — Uncertainty Quantification, OOD Detection, and Explainability Engine
=====================================================================================
Provides mathematical rigor for trustworthy probabilistic decision-support:
1. Shannon Entropy & Credible Margins: Measures aleatoric and epistemic confidence.
2. Out-of-Distribution (OOD) Detection: Mahalanobis distance against empirical training distribution.
3. Bayesian Evidence Trail: Prior → Evidence Likelihood Updates → Final Posterior distribution.
4. Feature Attribution & Sensitivity: Quantifies parameter risk drivers and counterfactual impact.
5. Safety Guardrails & Insufficient Evidence: Prevents overconfident predictions on anomalous inputs.
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

from backend.bayes.network import infer, PRIOR_MARGINALS, RISK_STATES

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA = ROOT / 'backend/data/processed/train_stations.csv'
FEATURES = ['ph', 'conductance', 'dissolved_oxygen', 'temperature']


class OODDetector:
    """
    Multivariate Mahalanobis Out-of-Distribution (OOD) Detector.
    Learns empirical mean and regularized covariance from the training split.
    """
    def __init__(self):
        self.mean_vector: np.ndarray = np.array([0.5, 0.2, 0.7, 0.4])
        self.inv_cov: np.ndarray = np.eye(len(FEATURES))
        self.ood_threshold: float = 13.28  # Chi-squared critical value df=4, p=0.99
        self._fitted = False
        self._fit_from_train_data()

    def _fit_from_train_data(self):
        if TRAIN_DATA.exists():
            df = pd.read_csv(TRAIN_DATA)
            X = df[FEATURES].dropna().to_numpy()
            self.mean_vector = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            # Add small Tikhonov regularization for numerical stability
            cov += np.eye(len(FEATURES)) * 1e-5
            self.inv_cov = np.linalg.pinv(cov)

            # Compute empirical Mahalanobis distances on training set
            dists = [mahalanobis(x, self.mean_vector, self.inv_cov) for x in X]
            # Set threshold at 99.5th empirical percentile or chi2 critical
            self.ood_threshold = float(np.percentile(dists, 99.5))
            self._fitted = True

    def evaluate(self, record: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates whether a sample is in-distribution or anomalous.
        """
        x = np.array([float(record.get(f, 0.5)) for f in FEATURES])

        # Physical boundary checks (Normalized scale [0, 1])
        boundary_violations = []
        for f in FEATURES:
            val = float(record.get(f, 0.5))
            if val < -0.05 or val > 1.05:
                boundary_violations.append(f"{f} ({val}) exceeds physical sensor domain [0, 1]")

        dist = float(mahalanobis(x, self.mean_vector, self.inv_cov))
        is_ood = dist > self.ood_threshold or len(boundary_violations) > 0

        # Severity rating
        if len(boundary_violations) > 0:
            severity = 'severe_violation'
        elif dist > self.ood_threshold * 1.5:
            severity = 'extreme_outlier'
        elif dist > self.ood_threshold:
            severity = 'moderate_outlier'
        else:
            severity = 'in_distribution'

        return {
            'is_out_of_distribution': is_ood,
            'mahalanobis_distance': round(dist, 3),
            'threshold': round(self.ood_threshold, 3),
            'severity': severity,
            'boundary_violations': boundary_violations,
            'guidance': (
                "Sensor observation deviates significantly from historical water profiles. "
                "Predictions carry heightened uncertainty. Field verification is advised."
                if is_ood else "Observation aligns with historical hydrological profile."
            )
        }


# Global OOD detector instance
OOD_DETECTOR = OODDetector()


def compute_uncertainty_metrics(probs_dict: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes Shannon entropy, Gini impurity, prediction margin, and confidence rating.
    """
    probs = np.array([max(1e-9, float(probs_dict.get(s, 0.0))) for s in RISK_STATES])
    probs /= probs.sum()  # ensure normalized

    k = len(probs)
    # Normalized Shannon Entropy in [0, 1]
    raw_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(k)
    normalized_entropy = float(raw_entropy / max_entropy)

    # Gini Impurity
    gini = float(1.0 - np.sum(probs ** 2))

    # Prediction Margin: Top-1 minus Top-2 probability
    sorted_probs = np.sort(probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1]) if k > 1 else 1.0

    top_prob = float(sorted_probs[0])

    # Wilson score / normal approximation standard error for top probability
    se = float(np.sqrt(top_prob * (1.0 - top_prob) / 100.0))
    ci_lower = max(0.0, top_prob - 1.96 * se)
    ci_upper = min(1.0, top_prob + 1.96 * se)

    # Trustworthiness / Confidence Tier
    if normalized_entropy < 0.35 and margin > 0.40:
        confidence_tier = 'High Confidence'
    elif normalized_entropy < 0.65 and margin > 0.15:
        confidence_tier = 'Moderate Confidence'
    else:
        confidence_tier = 'Low Confidence / Elevated Uncertainty'

    return {
        'normalized_entropy': round(normalized_entropy, 4),
        'gini_impurity': round(gini, 4),
        'prediction_margin': round(margin, 4),
        'top_class_confidence': round(top_prob, 4),
        'credible_interval_95': [round(ci_lower, 4), round(ci_upper, 4)],
        'confidence_tier': confidence_tier
    }


def compute_bayesian_evidence_trail(
    evidence: Dict[str, float],
    prior_marginals: Dict[str, Dict[str, float]],
    posterior_result: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Generates step-by-step Bayesian evidence trail:
    Prior beliefs -> Parameter Likelihood Updates -> Posterior marginals.
    """
    trail_steps = []

    # Prior baseline step
    risk_priors = prior_marginals.get('overall_risk', {})
    trail_steps.append({
        'step': 1,
        'title': 'Prior Belief Distribution',
        'type': 'prior',
        'description': 'Base distribution before observing parameter telemetry (derived from empirical training distribution).',
        'probabilities': risk_priors
    })

    # Individual evidence contributions
    step_num = 2
    for feat in FEATURES:
        if feat in evidence:
            val = float(evidence[feat])
            # Run single-evidence query to isolate feature's individual diagnostic power
            single_res = infer({feat: val})
            single_post = single_res['posterior']['overall_risk']

            # Calculate shift towards elevated risk (moderate, high, or critical)
            prior_elevated = sum(risk_priors.get(s, 0) for s in ['moderate', 'high', 'critical'])
            single_elevated = sum(single_post.get(s, 0) for s in ['moderate', 'high', 'critical'])
            shift = single_elevated - prior_elevated

            if shift > 0.10:
                direction = 'Substantially elevates risk'
            elif shift > 0.02:
                direction = 'Moderately elevates risk'
            elif shift < -0.10:
                direction = 'Substantially lowers risk (favorable indicator)'
            elif shift < -0.02:
                direction = 'Moderately lowers risk'
            else:
                direction = 'Neutral / Baseline consistency'

            trail_steps.append({
                'step': step_num,
                'title': f'Evidence: {feat.replace("_", " ").title()}',
                'type': 'evidence_update',
                'parameter': feat,
                'observed_value': round(val, 3),
                'shift_in_elevated_risk': round(shift, 4),
                'diagnostic_direction': direction,
                'marginal_risk_distribution': single_post
            })
            step_num += 1

    # Final combined posterior step
    final_post = posterior_result.get('overall_risk', {})
    trail_steps.append({
        'step': step_num,
        'title': 'Final Conditioned Posterior',
        'type': 'posterior',
        'description': 'Exact joint Bayesian marginal distribution conditioned on all provided telemetry.',
        'probabilities': final_post
    })

    return {
        'evidence_steps': trail_steps,
        'total_observed_parameters': len([f for f in FEATURES if f in evidence])
    }


def compute_feature_attributions(
    evidence: Dict[str, float],
    champion_model: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Computes sensitivity-based feature attribution:
    Evaluates impact of each parameter on elevated risk probability.
    """
    base_res = infer(evidence)
    base_post = base_res['posterior']['overall_risk']
    base_elevated = sum(base_post.get(s, 0.0) for s in ['moderate', 'high', 'critical'])

    attributions = []
    # Neutral reference values for normalized scale
    neutral_reference = {'ph': 0.50, 'conductance': 0.20, 'dissolved_oxygen': 0.85, 'temperature': 0.35}

    for feat in FEATURES:
        val = float(evidence.get(feat, neutral_reference[feat]))
        # Counterfactual: what if this feature were neutral/compliant?
        cf_evidence = dict(evidence)
        cf_evidence[feat] = neutral_reference[feat]

        cf_res = infer(cf_evidence)
        cf_elevated = sum(cf_res['posterior']['overall_risk'].get(s, 0.0) for s in ['moderate', 'high', 'critical'])

        # Impact = how much current value drives risk above neutral baseline
        raw_impact = abs(base_elevated - cf_elevated)
        signed_impact = base_elevated - cf_elevated  # positive = drives risk up

        attributions.append({
            'feature': feat,
            'label': feat.replace('_', ' ').title(),
            'observed_value': round(val, 3),
            'neutral_reference': neutral_reference[feat],
            'raw_impact': raw_impact,
            'signed_impact': round(signed_impact, 4),
            'impact_description': (
                "Primary Risk Driver" if signed_impact > 0.15 else
                ("Moderate Contaminant Factor" if signed_impact > 0.05 else
                 ("Protective Factor" if signed_impact < -0.05 else "Neutral"))
            )
        })

    # Normalize impact scores to percentage
    total_impact = sum(a['raw_impact'] for a in attributions) or 1.0
    for a in attributions:
        a['relative_importance_pct'] = round((a['raw_impact'] / total_impact) * 100.0, 1)

    attributions.sort(key=lambda a: a['relative_importance_pct'], reverse=True)
    return attributions


def explain_assessment(
    evidence: Dict[str, float],
    posterior_result: Dict[str, Dict[str, float]],
    champion_model: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Comprehensive explainability and uncertainty diagnostic report for a water assessment.
    """
    risk_probs = posterior_result.get('overall_risk', {})
    uncertainty = compute_uncertainty_metrics(risk_probs)
    ood = OOD_DETECTOR.evaluate(evidence)
    trail = compute_bayesian_evidence_trail(evidence, PRIOR_MARGINALS, posterior_result)
    attributions = compute_feature_attributions(evidence, champion_model)

    # Insufficient evidence guardrail
    observed_count = len([f for f in FEATURES if f in evidence and evidence[f] is not None])
    insufficient_evidence = observed_count < 2

    return {
        'uncertainty': uncertainty,
        'out_of_distribution': ood,
        'evidence_trail': trail,
        'feature_attributions': attributions,
        'guardrails': {
            'insufficient_evidence': insufficient_evidence,
            'missing_parameters': [f for f in FEATURES if f not in evidence],
            'warning_message': (
                "Insufficient sensor evidence: fewer than 2 primary parameters provided. "
                "Results rely heavily on background priors." if insufficient_evidence else None
            ),
            'safety_disclaimer': (
                "AQUA / BAYES is a statistical risk observatory for decision support. "
                "Critical alerts require immediate confirmatory laboratory water testing."
            )
        }
    }
