import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getReadiness, getModels, getDatasetStatus, listExperiments } from '../api/client';
import type { ReadinessResponse, ModelInfo, DatasetStatusResponse, ExperimentResponse } from '../types';

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`badge ${ok ? 'badge-green' : 'badge-red'}`}>
      <span className={`dot ${ok ? 'dot-green' : 'dot-red'}`} />
      {label}
    </span>
  );
}

export default function Dashboard() {
  const [ready, setReady] = useState<ReadinessResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [dataset, setDataset] = useState<DatasetStatusResponse | null>(null);
  const [experiments, setExperiments] = useState<ExperimentResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [r, m, d, e] = await Promise.all([
          getReadiness(),
          getModels(),
          getDatasetStatus(),
          listExperiments(),
        ]);
        setReady(r);
        setModels(m);
        setDataset(d);
        setExperiments(e);
      } catch (err) {
        setError('Cannot connect to backend. Make sure the API server is running on http://127.0.0.1:8000');
      }
    };
    load();
  }, []);

  if (error) {
    return (
      <div>
        <h1 className="page-title">Dashboard</h1>
        <div className="alert alert-error">{error}</div>
        <div className="card">
          <p className="text-sm text-muted">Start the backend with:</p>
          <pre>python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000</pre>
        </div>
      </div>
    );
  }

  const activeModel = models.find(m => m.is_active);
  const recentExps = experiments.slice(0, 5);

  return (
    <div>
      <h1 className="page-title">System Dashboard</h1>
      <p className="page-subtitle">
        Quantum-Centric Federated Learning — Screen-Time Behavioural Risk Assessment (Research Prototype)
      </p>

      {/* System Readiness */}
      <div className="card">
        <div className="card-title">System Readiness</div>
        {ready ? (
          <>
            <div className="flex gap-2 flex-wrap mt-1">
              <StatusBadge ok={ready.api_operational} label="API" />
              <StatusBadge ok={ready.database_operational} label="Database" />
              <StatusBadge ok={ready.model_available} label="Model Ready" />
              <StatusBadge ok={ready.quantum_available} label="Quantum (PennyLane)" />
            </div>
            <p className="text-sm mt-2" style={{ color: ready.ready ? '#6ee7b7' : '#fcd34d' }}>
              {ready.message}
            </p>
            {!ready.model_available && (
              <div className="alert alert-warning mt-2">
                No trained model is registered. Run a <strong>QAOA+VQC experiment</strong> from the Experiments page to train one.
              </div>
            )}
          </>
        ) : (
          <p className="text-muted text-sm"><span className="spin">⟳</span> Loading...</p>
        )}
      </div>

      <div className="grid-3">
        {/* Dataset */}
        <div className="card">
          <div className="card-title">Dataset</div>
          {dataset ? (
            <>
              <div className="stat-value">{dataset.total_samples}</div>
              <div className="stat-label">Labelled samples</div>
              {dataset.dataset_available ? (
                <span className="badge badge-green mt-2">Available</span>
              ) : (
                <span className="badge badge-yellow mt-2">Not configured</span>
              )}
              <p className="text-sm text-muted mt-2">{dataset.message}</p>
            </>
          ) : (
            <p className="text-muted text-sm"><span className="spin">⟳</span></p>
          )}
        </div>

        {/* Active Model */}
        <div className="card">
          <div className="card-title">Active Model</div>
          {activeModel ? (
            <>
              <div className="stat-value" style={{ fontSize: '1.1rem' }}>{activeModel.model_type}</div>
              <div className="stat-label">v{activeModel.version}</div>
              {activeModel.metrics?.best_val_accuracy !== undefined && (
                <p className="text-sm mt-2">
                  Val accuracy: <strong>{(Number(activeModel.metrics.best_val_accuracy) * 100).toFixed(1)}%</strong>
                  <span className="text-muted"> (synthetic data)</span>
                </p>
              )}
            </>
          ) : (
            <>
              <div className="stat-value" style={{ fontSize: '1rem', color: '#64748b' }}>None</div>
              <p className="text-sm text-muted mt-1">
                No trained model available.
              </p>
            </>
          )}
        </div>

        {/* Experiments */}
        <div className="card">
          <div className="card-title">Experiments</div>
          <div className="stat-value">{experiments.length}</div>
          <div className="stat-label">Total experiments</div>
          <Link to="/experiments" className="btn btn-primary btn-sm mt-2">
            View All
          </Link>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <div className="card-title">Quick Actions</div>
        <div className="flex gap-2 flex-wrap">
          <Link to="/camera" className="btn btn-primary">🎥 Record Video</Link>
          <Link to="/upload" className="btn btn-secondary">⬆ Upload Video</Link>
          <Link to="/experiments" className="btn btn-secondary">🔬 Run Experiment</Link>
        </div>
      </div>

      {/* Recent Experiments */}
      {recentExps.length > 0 && (
        <div className="card">
          <div className="card-title">Recent Experiments</div>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {recentExps.map(exp => (
                <tr key={exp.experiment_id}>
                  <td>{exp.name}</td>
                  <td><span className="badge badge-purple">{exp.experiment_type}</span></td>
                  <td>
                    <span className={`badge ${
                      exp.status === 'completed' ? 'badge-green' :
                      exp.status === 'running' ? 'badge-yellow' :
                      exp.status === 'failed' ? 'badge-red' : 'badge-gray'
                    }`}>{exp.status}</span>
                  </td>
                  <td className="text-muted">{new Date(exp.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Disclaimer */}
      <div className="alert alert-warning">
        <strong>Research Prototype Disclaimer:</strong> This system is an experimental research tool only.
        Predictions are NOT clinical diagnoses and must not be used for any medical or clinical decision-making.
        No real labelled dataset is included. Results shown are from synthetic data experiments only.
      </div>
    </div>
  );
}
