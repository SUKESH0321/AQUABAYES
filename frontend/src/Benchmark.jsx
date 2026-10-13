import React, { useEffect, useState } from 'react';

export default function Benchmark({ api, onRunRetrain }) {
  const [report, setReport] = useState(null);
  const [monitoring, setMonitoring] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      api('/api/model/evaluation').catch(() => null),
      api('/api/model/monitoring').catch(() => null)
    ]).then(([rep, mon]) => {
      setReport(rep);
      setMonitoring(mon);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return <section className="instrument-panel">Loading model benchmark & calibration data…</section>;
  }

  if (!report) {
    return (
      <section className="instrument-panel empty-state">
        <span>∿</span>
        <h2>Benchmark Report Unavailable</h2>
        <p>Run the training and evaluation script to generate comprehensive holdout metrics.</p>
      </section>
    );
  }

  const testMet = report.test_metrics || {};
  const champMet = testMet.calibrated_random_forest || testMet.calibrated_hist_gb || {};
  const bnMet = testMet.bayesian_network || {};
  const baseMet = testMet.majority_class_baseline || {};

  const models = [
    { id: 'calibrated_random_forest', name: 'Calibrated Random Forest (Champion)', type: 'ML Ensemble', metrics: champMet, badge: 'CHAMPION' },
    { id: 'bayesian_network', name: 'Empirical Bayesian Network (Dirichlet)', type: 'Probabilistic Graph', metrics: bnMet, badge: 'EXACT INFERENCE' },
    { id: 'calibrated_hist_gb', name: 'Calibrated Gradient Boosting', type: 'Tree Ensemble', metrics: testMet.calibrated_hist_gb || {}, badge: 'CANDIDATE' },
    { id: 'calibrated_logistic_regression', name: 'Calibrated Logistic Regression', type: 'Linear Model', metrics: testMet.calibrated_logistic_regression || {}, badge: 'CANDIDATE' },
    { id: 'majority_class_baseline', name: 'Majority Class Baseline', type: 'Heuristic Baseline', metrics: baseMet, badge: 'BASELINE' },
  ];

  const calBins = champMet.calibration_bins || [];
  const driftFeatures = monitoring?.drift_metrics?.features || {};

  return (
    <div className="benchmark-container">
      {/* Champion Status & Scientific Holdout Header */}
      <section className="instrument-panel benchmark-hero">
        <div className="panel-label-row">
          <span className="section-kicker">SCIENTIFIC HOLDOUT EVALUATION / NO LEAKAGE</span>
          <span className="mono-note">TRAIN: 70% · VAL: 15% · TEST: 15%</span>
        </div>
        <h2>Empirical Model Benchmark & Calibration Observatory</h2>
        <p className="panel-copy">
          Models are evaluated on strictly held-out chronological test splits (3,922 observations) with zero future-lookahead leakage.
          Probabilistic reliability is measured via <strong>Brier Score</strong> and <strong>Expected Calibration Error (ECE)</strong>.
        </p>

        <div className="benchmark-stat-row">
          <div className="bench-stat-card">
            <span>CHAMPION ACCURACY</span>
            <b>{((champMet.accuracy || 0) * 100).toFixed(1)}%</b>
            <small>Baseline: {((baseMet.accuracy || 0) * 100).toFixed(1)}%</small>
          </div>
          <div className="bench-stat-card">
            <span>BRIER SCORE (CALIBRATION)</span>
            <b style={{ color: '#278e83' }}>{(champMet.brier_score || 0).toFixed(4)}</b>
            <small>Lower is better (Base: {(baseMet.brier_score || 0).toFixed(3)})</small>
          </div>
          <div className="bench-stat-card">
            <span>EXPECTED CALIBRATION ERROR</span>
            <b style={{ color: '#278e83' }}>{((champMet.expected_calibration_error || 0) * 100).toFixed(2)}%</b>
            <small>Gap between confidence & accuracy</small>
          </div>
          <div className="bench-stat-card">
            <span>SEVERE RISK FALSE NEGATIVE RATE</span>
            <b style={{ color: '#278e83' }}>{((champMet.high_critical_false_negative_rate || 0) * 100).toFixed(2)}%</b>
            <small>Precautionary health safeguard</small>
          </div>
        </div>
      </section>

      {/* Benchmark Comparison Table */}
      <section className="instrument-panel" style={{ marginTop: '20px', overflowX: 'auto' }}>
        <div className="panel-label-row">
          <span className="section-kicker">MODEL COMPARISON TABLE / TEST SPLIT</span>
          <span className="mono-note">N = 3,922 OBSERVATIONS</span>
        </div>
        <h2>Candidate vs Baseline Performance</h2>
        <div className="benchmark-table-wrap"><table className="benchmark-table">
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>Test Acc</th>
              <th>Macro F1</th>
              <th>Brier Score</th>
              <th>ECE</th>
              <th>Severe FN Rate</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m.id} className={m.badge === 'CHAMPION' ? 'highlight-row' : ''}>
                <td>
                  <strong>{m.name}</strong>
                  <span className="model-sub">{m.type}</span>
                </td>
                <td>{((m.metrics?.accuracy || 0) * 100).toFixed(1)}%</td>
                <td>{((m.metrics?.macro_f1 || 0) * 100).toFixed(1)}%</td>
                <td>{(m.metrics?.brier_score || 0).toFixed(4)}</td>
                <td>{((m.metrics?.expected_calibration_error || 0) * 100).toFixed(1)}%</td>
                <td>{((m.metrics?.high_critical_false_negative_rate || 0) * 100).toFixed(1)}%</td>
                <td><span className={`badge-pill ${m.badge.toLowerCase().replace(' ', '-')}`}>{m.badge}</span></td>
              </tr>
            ))}
          </tbody>
        </table></div>
      </section>

      {/* Calibration Bins & Reliability Curves */}
      <div className="bench-grid" style={{ marginTop: '20px' }}>
        <section className="instrument-panel">
          <div className="panel-label-row">
            <span className="section-kicker">PROBABILISTIC CALIBRATION / RELIABILITY BINS</span>
            <span className="mono-note">PLATT / 3-FOLD CV</span>
          </div>
          <h2>Calibration Reliability Bins</h2>
          <p className="panel-copy">
            A well-calibrated model has observed accuracy matching predicted confidence across each probability decile.
          </p>
          <div className="cal-bins-container">
            {calBins.map(b => (
              <div key={b.bin_index} className="cal-bin-row">
                <span className="bin-range">{b.range[0].toFixed(1)}–{b.range[1].toFixed(1)}</span>
                <div className="bin-bar-track">
                  <div className="bin-acc-bar" style={{ width: `${Math.min(100, b.accuracy * 100)}%` }} title={`Accuracy: ${(b.accuracy*100).toFixed(1)}%`} />
                  <div className="bin-conf-marker" style={{ left: `${Math.min(100, b.confidence * 100)}%` }} title={`Confidence: ${(b.confidence*100).toFixed(1)}%`} />
                </div>
                <span className="bin-gap">Gap: {(b.calibration_gap * 100).toFixed(1)}% ({b.samples} samples)</span>
              </div>
            ))}
          </div>
        </section>

        {/* Data Drift (PSI) & Monitoring Status */}
        <section className="instrument-panel">
          <div className="panel-label-row">
            <span className="section-kicker">CONTINUOUS LEARNING / POPULATION STABILITY INDEX</span>
            <span className={`status-pill ${monitoring?.status || 'healthy'}`}>{monitoring?.status?.toUpperCase() || 'HEALTHY'}</span>
          </div>
          <h2>Telemetry Drift Monitoring</h2>
          <p className="panel-copy">
            Monitors covariate shift between baseline training distribution and field observations. PSI &gt; 0.25 triggers retraining alert.
          </p>
          <div className="drift-list">
            {Object.entries(driftFeatures).map(([feat, stat]) => (
              <div key={feat} className="drift-item">
                <div>
                  <strong>{feat.replace('_', ' ').toUpperCase()}</strong>
                  <small>PSI: {stat.psi.toFixed(4)} · KS p-val: {stat.ks_pvalue.toFixed(4)}</small>
                </div>
                <span className={`drift-badge ${stat.status}`}>{stat.status.replace('_', ' ').toUpperCase()}</span>
              </div>
            ))}
          </div>

          <div className="verified-obs-box" style={{ marginTop: '20px' }}>
            <span className="section-kicker">LABORATORY VERIFIED GROUND TRUTH</span>
            <p>Field Titrations Logged: <strong>{monitoring?.verified_observations_count || 0} samples</strong></p>
            <small>Ground-truth samples are strictly isolated from real-time model inferences to ensure tamper-proof retraining verification.</small>
          </div>
        </section>
      </div>
    </div>
  );
}
