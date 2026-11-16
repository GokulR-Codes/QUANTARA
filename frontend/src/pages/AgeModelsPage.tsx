import { useState, useEffect } from 'react';
import { getAgeGroupStatuses } from '../api/authClient';
import type { AgeGroupModelStatus } from '../types/auth';

const GROUP_ICONS: Record<string, string> = {
  '5_7': '🧒',
  '7_10': '👦',
  '10_12': '🧑',
};

const GROUP_RESEARCH_BADGE: Record<string, string> = {
  '5_7': 'badge-blue',
  '7_10': 'badge-blue',
  '10_12': 'badge-blue',
};

export default function AgeModelsPage() {
  const [statuses, setStatuses] = useState<AgeGroupModelStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAgeGroupStatuses()
      .then(setStatuses)
      .catch(() => setError('Failed to load age group model statuses. Are you logged in?'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="page-title">🧬 Age-Specific Models</h1>
      <p className="page-subtitle">
        Q-CARE uses separate QAOA–VQC model configurations for each child age group.
        Models trained on each group are stored independently and are never substituted across groups.
      </p>

      <div className="alert alert-info">
        <strong>Research Note:</strong> The processing parameters below reflect the developmental and behavioural
        literature for each age band. Configuration values marked "(default)" are research-informed starting
        points — they will be refined as labelled training data becomes available for each age group.
      </div>

      {loading && (
        <div className="card">
          <span className="spin">⟳</span> Loading age group statuses…
        </div>
      )}

      {error && <div className="alert alert-error">{error}</div>}

      {!loading && !error && (
        <div className="grid-3" style={{ gap: '1rem' }}>
          {statuses.map((model) => (
            <div
              key={model.age_group}
              className="card"
              style={{
                borderLeft: model.has_active_model
                  ? '3px solid #22c55e'
                  : '3px solid #64748b',
              }}
            >
              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '1.75rem' }}>{GROUP_ICONS[model.age_group] ?? '👤'}</span>
                <div>
                  <div className="card-title" style={{ marginBottom: 0 }}>{model.label}</div>
                  <code style={{ fontSize: '0.75rem', color: '#94a3b8' }}>group: {model.age_group}</code>
                </div>
              </div>

              {/* Model availability */}
              <div style={{ marginBottom: '0.75rem' }}>
                {model.has_active_model ? (
                  <span className="badge badge-green">✓ Trained model available</span>
                ) : (
                  <span className="badge badge-gray">No trained model — using default config</span>
                )}
                {model.is_approved && (
                  <span className="badge badge-green" style={{ marginLeft: '0.25rem' }}>Approved</span>
                )}
              </div>

              {model.model_id && (
                <p className="text-sm text-muted">
                  <strong>Model ID:</strong> <code>{model.model_id}</code>
                  {model.model_version && <> · v{model.model_version}</>}
                </p>
              )}

              {/* Processing config */}
              {model.processing_config_summary && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div className="text-sm" style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                    Processing Configuration
                  </div>
                  <table style={{ fontSize: '0.78rem', width: '100%' }}>
                    <tbody>
                      {Object.entries(model.processing_config_summary).map(([k, v]) => (
                        <tr key={k}>
                          <td className="text-muted" style={{ paddingRight: '0.5rem', whiteSpace: 'nowrap' }}>
                            {k.replace(/_/g, ' ')}
                          </td>
                          <td>
                            <code>{String(v)}</code>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Validation metrics */}
              {model.validation_metrics && Object.keys(model.validation_metrics).length > 0 && (
                <div style={{ marginTop: '0.75rem' }}>
                  <div className="text-sm" style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                    Validation Metrics
                  </div>
                  <table style={{ fontSize: '0.78rem', width: '100%' }}>
                    <tbody>
                      {Object.entries(model.validation_metrics).map(([k, v]) => (
                        <tr key={k}>
                          <td className="text-muted" style={{ paddingRight: '0.5rem' }}>
                            {k.replace(/_/g, ' ')}
                          </td>
                          <td><code>{String(v)}</code></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Research notes */}
              {model.research_notes && (
                <div
                  style={{
                    marginTop: '0.75rem',
                    padding: '0.5rem',
                    background: 'rgba(99,102,241,0.08)',
                    borderRadius: '4px',
                    fontSize: '0.78rem',
                  }}
                >
                  <span className={`badge ${GROUP_RESEARCH_BADGE[model.age_group] ?? 'badge-blue'}`} style={{ marginBottom: '0.35rem' }}>
                    Research basis
                  </span>
                  <p style={{ margin: 0, color: '#94a3b8', lineHeight: 1.5 }}>{model.research_notes}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-title">About Age-Adaptive Processing</div>
        <p className="text-sm text-muted">
          Each age group uses different sequence lengths, minimum tracked frames, QAOA circuit depths, and VQC
          configurations. These parameters are based on the expected behavioural complexity and attention span
          characteristics for each developmental stage. A model trained on one age group is <strong>never
          substituted</strong> for another — if no trained model exists for a group, the system returns{' '}
          <code>model_not_ready</code> rather than producing an invalid result.
        </p>
        <p className="text-sm text-muted mt-1">
          Age categories are <strong>always guardian-selected</strong> — the system never attempts to infer
          age from video, appearance, or biometric data.
        </p>
      </div>
    </div>
  );
}
