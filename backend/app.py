from pathlib import Path
import json
from flask import Flask, jsonify, request, send_from_directory
from .db import init_db, create_assessment, list_assessments, get_assessment, list_sources, add_source, update_source
from .bayes.network import infer, network_metadata, RISK_STATES
from .bayes.metadata import FEATURE_LABELS, load_metadata, feature_stats
from .data.standards import calculate_wqi, classify_risk_state, STANDARDS_METADATA
from .decision.influence import decide, metadata as decision_metadata
from .ml.classifiers import CalibratedRiskModel, WQIRegressor, ARTIFACTS_DIR
from .ml.explainability import explain_assessment
from .ml.monitoring import check_telemetry_drift, log_verified_observation, get_verified_observations_count, get_active_model_info

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / 'frontend/dist'
EVAL_REPORT_PATH = ROOT / 'backend/data/processed/evaluation_report.json'

app = Flask(__name__, static_folder=str(FRONTEND / 'assets'), static_url_path='/assets')
init_db()

DISCLAIMER = (
    'This system is an AI-based decision-support and educational observatory. '
    'Its probabilistic predictions are grounded in hydrological standards (WHO / BIS IS 10500) '
    'and empirical Bayesian inference. Predictions do not replace accredited laboratory testing '
    'or statutory public-health directives. High or critical risk alerts warrant immediate confirmatory testing.'
)

# Load Champion ML Model and WQI Regressor if serialized artifacts exist
CHAMPION_MODEL = None
CHAMPION_PATH = ARTIFACTS_DIR / 'champion_risk_model.pkl'
if CHAMPION_PATH.exists():
    try:
        CHAMPION_MODEL = CalibratedRiskModel.load(CHAMPION_PATH)
    except Exception as e:
        print(f"Warning: Could not load champion model: {e}")

WQI_MODEL = None
WQI_PATH = ARTIFACTS_DIR / 'wqi_regressor.pkl'
if WQI_PATH.exists():
    try:
        WQI_MODEL = WQIRegressor.load(WQI_PATH)
    except Exception as e:
        print(f"Warning: Could not load WQI regressor: {e}")


def validate(data):
    required = list(FEATURE_LABELS)
    errors = {}
    for key in required:
        if key not in data or data[key] in ('', None):
            errors[key] = 'Required supported normalized index (0–1).'
        else:
            try:
                value = float(data[key])
                if not 0.0 <= value <= 1.0:
                    errors[key] = 'Must be between 0 and 1 for the source-normalized index.'
            except (TypeError, ValueError):
                errors[key] = 'Must be numeric.'
    if errors:
        return None, errors
    return {k: float(data[k]) for k in required}, None


def assessment_payload(data, persist=False):
    evidence, errors = validate(data.get('evidence', data))
    if errors:
        return None, (jsonify({'errors': errors}), 400)

    # 1. Exact Bayesian Network Inference
    bayes_res = infer(evidence)
    posterior = bayes_res['posterior']

    # 2. Decision Influence and Expected Utility
    decision = decide(posterior)

    # 3. Standards & WQI Calculation
    wqi_score = round(calculate_wqi(evidence), 2)
    standards_class = classify_risk_state(wqi_score, evidence)

    # 4. Champion Machine Learning Model Inference (if available)
    ml_prediction = None
    if CHAMPION_MODEL is not None:
        try:
            ml_prediction = CHAMPION_MODEL.predict_record(evidence)
        except Exception:
            pass

    # 5. Explainability & Uncertainty Quantification
    explanation = explain_assessment(evidence, posterior, CHAMPION_MODEL)

    active_model = get_active_model_info()

    payload = {
        'sample_date': data.get('sample_date') or 'Today',
        'source_id': data.get('source_id'),
        'evidence': evidence,
        'states': bayes_res['states'],
        'posterior': posterior,
        'prior_marginals': bayes_res.get('prior_marginals', {}),
        'evidence_shifts': bayes_res.get('evidence_shifts', {}),
        'utilities': decision['utilities'],
        'recommendation': decision['recommendation'],
        'factors': explanation['feature_attributions'],
        'wqi': wqi_score,
        'standards_classification': standards_class,
        'champion_ml': ml_prediction,
        'uncertainty': explanation['uncertainty'],
        'out_of_distribution': explanation['out_of_distribution'],
        'evidence_trail': explanation['evidence_trail'],
        'guardrails': explanation['guardrails'],
        'model_version': active_model.get('version_id', 'v1.0.0-champion'),
        'active_model_info': active_model
    }

    if persist:
        payload['id'] = create_assessment(payload)

    return {
        **payload,
        'recommended_action': payload['recommendation'],
        'recommendation_label': decision['recommendation_label'],
        'dominant_risk': decision['dominant_risk'],
        'policy': decision['policy'],
        'disclaimer': DISCLAIMER
    }, None


@app.get('/api/health')
def health():
    return jsonify({
        'ok': True,
        'service': 'aqua-bayes-intelligence',
        'model_version': 'v1.0.0-champion',
        'ml_engine': 'Calibrated Ensemble + Exact VariableElimination Bayesian Network',
        'standards': ['WHO_GDWQ_2022', 'BIS_IS_10500_2012', 'CPCB_DESIGNATED_BEST_USE']
    })


@app.get('/api/model/network')
def model_network():
    return jsonify(network_metadata())


@app.get('/api/model/decision-network')
def model_decision():
    return jsonify(decision_metadata())


@app.get('/api/model/about')
def model_about():
    return jsonify({
        'metadata': load_metadata(),
        'feature_stats': feature_stats(),
        'feature_labels': FEATURE_LABELS,
        'standards_metadata': STANDARDS_METADATA,
        'active_model': get_active_model_info(),
        'disclaimer': DISCLAIMER
    })


@app.get('/api/model/evaluation')
def model_evaluation():
    if EVAL_REPORT_PATH.exists():
        data = json.loads(EVAL_REPORT_PATH.read_text())
        return jsonify(data)
    return jsonify({'error': 'Evaluation report not yet generated'}), 404


@app.get('/api/model/monitoring')
def model_monitoring():
    drift = check_telemetry_drift([])
    active = get_active_model_info()
    verified_count = get_verified_observations_count()
    return jsonify({
        'active_model': active,
        'drift_metrics': drift,
        'verified_observations_count': verified_count,
        'status': drift.get('status', 'healthy')
    })


@app.post('/api/model/verified-observations')
def record_verified_observation():
    data = request.get_json(force=True)
    required = ['ph', 'conductance', 'dissolved_oxygen', 'temperature']
    if not all(k in data for k in required):
        return jsonify({'error': f'Required fields: {required}'}), 400
    new_id = log_verified_observation(data)
    return jsonify({'success': True, 'id': new_id}), 201


@app.get('/api/sources')
def sources():
    return jsonify(list_sources())


@app.post('/api/sources')
def source_create():
    data = request.get_json(force=True)
    if not data.get('name'):
        return jsonify({'error': 'Source name is required'}), 400
    return jsonify({'id': add_source(data)}), 201


@app.patch('/api/sources/<int:source_id>')
def source_update(source_id):
    data = request.get_json(force=True)
    if not data.get('name'):
        return jsonify({'error': 'Source name is required'}), 400
    row = update_source(source_id, data)
    return (jsonify(row), 200) if row else (jsonify({'error': 'Source not found'}), 404)


@app.post('/api/assessments')
def assessments_create():
    payload, error = assessment_payload(request.get_json(force=True), True)
    return error if error else jsonify(payload), 201


@app.post('/api/assessments/what-if')
def what_if():
    payload, error = assessment_payload(request.get_json(force=True), False)
    return error if error else jsonify(payload)


@app.get('/api/assessments/<int:assessment_id>')
def assessment_detail(assessment_id):
    row = get_assessment(assessment_id)
    if not row:
        return jsonify({'error': 'Assessment not found'}), 404
    row['disclaimer'] = DISCLAIMER
    row['recommended_action'] = row['recommendation']
    row['recommendation_label'] = row['recommendation'].replace('_', ' ').title()
    row['dominant_risk'] = max(row['posterior']['overall_risk'], key=row['posterior']['overall_risk'].get)
    row['policy'] = decision_metadata()['utility_table']
    return jsonify(row)


@app.get('/api/history')
def history():
    source_id = request.args.get('source', type=int)
    return jsonify(list_assessments(int(request.args.get('limit', 100)), source_id, request.args.get('from'), request.args.get('to')))


@app.get('/api/dashboard')
def dashboard():
    rows = list_assessments(250)
    counts = {k: 0 for k in RISK_STATES}
    actions = {}
    for row in rows:
        post = row.get('posterior', {}).get('overall_risk', {})
        if post:
            risk = max(post, key=post.get)
            counts[risk] = counts.get(risk, 0) + 1
        act = row.get('recommendation', 'maintain_monitoring')
        actions[act] = actions.get(act, 0) + 1

    monitoring_info = check_telemetry_drift([])

    return jsonify({
        'counts': counts,
        'samples_analyzed': len(rows),
        'actions': actions,
        'recent': rows[:6],
        'latest': rows[0] if rows else None,
        'active_model': get_active_model_info(),
        'monitoring': monitoring_info
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def spa(path):
    if path.startswith('api/'):
        return jsonify({'error': 'Not found'}), 404
    target = FRONTEND / path
    if target.exists() and target.is_file():
        return send_from_directory(FRONTEND, path)
    return send_from_directory(FRONTEND, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(__import__('os').environ.get('PORT', 3000)))
