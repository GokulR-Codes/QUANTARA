import { useState, useEffect, useCallback } from 'react';
import { createExperiment, listExperiments } from '../api/client';
import type { ExperimentResponse } from '../types';

const EXP_TYPES = [
  {
    id: 'classical',
    label: 'Classical Baseline',
    desc: 'Train an MLP on synthetic data. Quick comparison baseline.',
    icon: '🖥',
  },
  {
    id: 'qaoa_vqc',
    label: 'QAOA + VQC (Quantum)',
    desc: 'Run QAOA feature selection then train the VQC. Registers a model checkpoint.',
    icon: '⚛',
  },
  {
    id: 'federated',
    label: 'Federated Learning',
    desc: 'Simulated FedAvg across synthetic client partitions.',
    icon: '🌐',
  },
];

function StatusBadge({ status }: { status: string }) {
  const cls = {
    completed: 'badge-green',
    running: 'badge-yellow',
    failed: 'badge-red',
    queued: 'badge-blue',
  }[status] ?? 'badge-gray';
  return <span className={`badge ${cls}`}>{status}</span>;
}

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([]);
  const [creating, setCreating] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<ExperimentResponse | null>(null);

  const loadExperiments = useCallback(async () => {
    try {
      const list = await listExperiments();
      setExperiments(list);
      // Update selected if it's in the list
      if (selected) {
        const updated = list.find(e => e.experiment_id === selected.experiment_id);
        if (updated) setSelected(updated);
      }
    } catch { /* ignore */ }
  }, [selected]);

  useEffect(() => {
    loadExperiments();
    const interval = setInterval(loadExperiments, 3000);
    return () => clearInterval(interval);
  }, [loadExperiments]);

  const handleCreate = async (type: string) => {
    setCreating(type);
    setError(null);
    try {
      const exp = await createExperiment(
        `${type} experiment ${new Date().toLocaleTimeString()}`,
        type,
        { n_qubits: 4, qaoa_depth: 2, vqc_depth: 2, epochs: 5 },
      );
      setExperiments(prev => [exp, ...prev]);
      setSelected(exp);
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? 'Failed to create experiment');
    } finally {
      setCreating(null);
    }
  };

  const formatResults = (results: Record<string, unknown>): string => JSON.stringify(results, null, 2);

  return (
    <div>
      <h1 className="page-title">Experiments</h1>
      <p className="page-subtitle">
        Run quantum, classical, or federated experiments on synthetic data.
        All results are labelled as synthetic — real data requires a labelled dataset.
      </p>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="alert alert-warning">
        <strong>Synthetic Data Only:</strong> These experiments run on randomly generated data.
        Results do not represent real behavioural risk assessments.
        The QAOA+VQC experiment registers a checkpoint trained on random synthetic features; it is listed under
        Settings as <strong>not usable for video inference</strong> (video inference needs a pipeline bundle with
        trained CNN/LSTM/fusion weights, a QAOA mapping over the fused features, and a compatible VQC).
      </div>

      {/* Experiment types */}
      <div className="grid-3">
        {EXP_TYPES.map(({ id, label, desc, icon }) => (
          <div key={id} className="card">
            <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>{icon}</div>
            <div className="card-title">{label}</div>
            <p className="text-sm text-muted">{desc}</p>
            <button
              className="btn btn-primary btn-sm mt-2"
              onClick={() => handleCreate(id)}
              disabled={creating !== null}
            >
              {creating === id ? <><span className="spin">⟳</span> Starting…</> : '▶ Run'}
            </button>
          </div>
        ))}
      </div>

      {/* Experiment list */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="card-title">All Experiments</div>
          <button className="btn btn-secondary btn-sm" onClick={loadExperiments}>↻ Refresh</button>
        </div>
        {experiments.length === 0 ? (
          <p className="text-muted text-sm">No experiments yet. Run one above.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map(exp => (
                <tr
                  key={exp.experiment_id}
                  style={{ cursor: 'pointer', background: selected?.experiment_id === exp.experiment_id ? '#252843' : '' }}
                  onClick={() => setSelected(exp)}
                >
                  <td>{exp.name}</td>
                  <td><span className="badge badge-purple">{exp.experiment_type}</span></td>
                  <td><StatusBadge status={exp.status} /></td>
                  <td className="text-muted">{new Date(exp.created_at).toLocaleString()}</td>
                  <td>
                    {exp.status === 'running' && <span className="spin text-sm">⟳</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Selected experiment details */}
      {selected && (
        <div className="card">
          <div className="card-title">
            {selected.name} — Details
            <span className="ml-2"><StatusBadge status={selected.status} /></span>
          </div>

          {selected.notes?.map((n, i) => (
            <div key={i} className="alert alert-info mt-1">{n}</div>
          ))}

          {selected.status === 'running' && (
            <div className="alert alert-info">
              <span className="spin">⟳</span> Experiment is running… This page auto-refreshes every 3 seconds.
            </div>
          )}

          {selected.results && (
            <>
              <div className="card-title mt-2">Results</div>
              {selected.results.disclaimer && (
                <div className="alert alert-warning">{String(selected.results.disclaimer)}</div>
              )}
              {selected.results.qaoa && (
                <div className="mt-2">
                  <strong>QAOA Output</strong>
                  <table className="mt-1">
                    <tbody>
                      <tr><td className="text-muted">Selected features</td>
                        <td>{(selected.results.qaoa as { selected_names: string[] }).selected_names?.join(', ')}</td>
                      </tr>
                      <tr><td className="text-muted">Objective value</td>
                        <td>{Number((selected.results.qaoa as { objective_value: number }).objective_value).toFixed(4)}</td>
                      </tr>
                      <tr><td className="text-muted">Optimization duration</td>
                        <td>{Number((selected.results.qaoa as { optimization_duration_s: number }).optimization_duration_s).toFixed(2)}s</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
              {selected.results.vqc != null && (
                <div className="mt-2">
                  <strong>VQC Training</strong>
                  <table className="mt-1">
                    <tbody>
                      <tr><td className="text-muted">Best val accuracy (synthetic)</td>
                        <td>{(Number((selected.results.vqc as { best_val_accuracy: number }).best_val_accuracy) * 100).toFixed(1)}%</td>
                      </tr>
                      <tr><td className="text-muted">Best epoch</td>
                        <td>{(selected.results.vqc as { best_epoch: number }).best_epoch}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
              {selected.results.final_accuracy !== undefined && (
                <p className="text-sm mt-2">
                  Final accuracy (synthetic): <strong>{(Number(selected.results.final_accuracy) * 100).toFixed(1)}%</strong>
                </p>
              )}
              {selected.results.checkpoint_path != null && (
                <p className="text-sm mt-1 text-muted">
                  Checkpoint: <code>{String(selected.results.checkpoint_path)}</code>
                </p>
              )}
            </>
          )}

          {selected.results && (
            <details className="mt-2">
              <summary className="text-sm text-muted">Raw results (JSON)</summary>
              <pre className="text-sm" style={{ overflowX: 'auto' }}>{formatResults(selected.results)}</pre>
            </details>
          )}

          {selected.status === 'failed' && selected.results?.error != null && (
            <div className="alert alert-error mt-2">
              <strong>Error:</strong> {String(selected.results.error)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
