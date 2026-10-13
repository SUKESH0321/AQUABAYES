import React from 'react';

export default function EvidenceTrail({ trail, uncertainty, ood, wqi, standardsClass }) {
  if (!trail || !trail.evidence_steps) {
    return null;
  }

  const steps = trail.evidence_steps;
  const colors = { low: '#65c6af', moderate: '#d4ae5e', high: '#d9854d', critical: '#c76660' };

  return (
    <div className="evidence-trail-container">
      {/* OOD and Guardrail Alert if applicable */}
      {ood && ood.is_out_of_distribution && (
        <div className="ood-alert-banner">
          <div className="ood-icon">⚠️</div>
          <div>
            <strong>OUT-OF-DISTRIBUTION OBSERVATION DETECTED</strong>
            <p>
              Mahalanobis Distance: <b>{ood.mahalanobis_distance}</b> (Threshold: {ood.threshold}).
              {ood.guidance}
            </p>
          </div>
        </div>
      )}

      {/* Uncertainty & Credible Summary Card */}
      {uncertainty && (
        <section className="instrument-panel uncertainty-summary-panel">
          <div className="panel-label-row">
            <span className="section-kicker">UNCERTAINTY QUANTIFICATION / METRIC AUDIT</span>
            <span className={`confidence-badge ${uncertainty.confidence_tier.toLowerCase().replace(/[\s\/]+/g, '-')}`}>
              {uncertainty.confidence_tier.toUpperCase()}
            </span>
          </div>
          <h2>Probabilistic Confidence & Credible Interval</h2>
          <div className="uncertainty-metric-grid">
            <div className="u-metric-item">
              <span>SHANNON ENTROPY</span>
              <b>{uncertainty.normalized_entropy}</b>
              <small>Scale [0, 1] · Lower = higher certainty</small>
            </div>
            <div className="u-metric-item">
              <span>PREDICTION MARGIN</span>
              <b>{((uncertainty.prediction_margin || 0) * 100).toFixed(1)}%</b>
              <small>Top-1 vs Top-2 class probability gap</small>
            </div>
            <div className="u-metric-item">
              <span>95% CREDIBLE INTERVAL</span>
              <b>
                {((uncertainty.credible_interval_95?.[0] || 0) * 100).toFixed(1)}% – {((uncertainty.credible_interval_95?.[1] || 0) * 100).toFixed(1)}%
              </b>
              <small>Wilson standard error bounds</small>
            </div>
            <div className="u-metric-item">
              <span>HORTON / BROWN WQI</span>
              <b style={{ color: '#278e83' }}>{wqi || '—'} / 100</b>
              <small>{standardsClass?.replace('_', ' ').toUpperCase() || 'STANDARDS INDEX'}</small>
            </div>
          </div>
        </section>
      )}

      {/* Step-by-Step Bayesian Evidence Trail */}
      <section className="instrument-panel" style={{ marginTop: '20px' }}>
        <div className="panel-label-row">
          <span className="section-kicker">BAYESIAN EVIDENCE TRAIL / SHIFT AUDIT</span>
          <span className="mono-note">{trail.total_observed_parameters} SENSORS OBSERVED</span>
        </div>
        <h2>Belief Progression: Prior → Telemetry → Posterior</h2>
        <p className="panel-copy">
          Follow the exact mathematical progression as each parameter observation updates prior beliefs into the joint posterior.
        </p>

        <div className="trail-timeline">
          {steps.map((step) => {
            const isPrior = step.type === 'prior';
            const isPost = step.type === 'posterior';
            const probs = step.probabilities || step.marginal_risk_distribution || {};

            return (
              <div key={step.step} className={`trail-step-card ${step.type}`}>
                <div className="step-badge-num">0{step.step}</div>
                <div className="step-content">
                  <div className="step-header">
                    <strong>{step.title}</strong>
                    {step.diagnostic_direction && (
                      <span className={`direction-tag ${step.shift_in_elevated_risk > 0 ? 'elevating' : 'lowering'}`}>
                        {step.diagnostic_direction}
                      </span>
                    )}
                  </div>
                  {step.description && <p className="step-desc">{step.description}</p>}
                  {step.observed_value !== undefined && (
                    <div className="step-observed-stat">
                      Observed Sensor Index: <b>{step.observed_value}</b> ·
                      Elevated Risk Shift: <b>{step.shift_in_elevated_risk > 0 ? '+' : ''}{(step.shift_in_elevated_risk * 100).toFixed(2)}%</b>
                    </div>
                  )}

                  <div className="step-prob-bar">
                    {['low', 'moderate', 'high', 'critical'].map((st) => {
                      const p = probs[st] || 0;
                      return (
                        <div key={st} className="prob-slice" style={{ width: `${Math.max(2, p * 100)}%`, background: colors[st] }} title={`${st}: ${(p * 100).toFixed(1)}%`}>
                          {p > 0.12 && <span>{(p * 100).toFixed(0)}%</span>}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
