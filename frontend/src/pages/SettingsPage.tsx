import { useState, useEffect } from 'react';
import { getReadiness, getModels } from '../api/client';
import type { ModelInfo } from '../types';

export default function SettingsPage() {
  const [ready, setReady] = useState<{ components: Array<{ name: string; status: string; detail?: string }> } | null>(null);

  const [models, setModels] = useState<ModelInfo[] | null>(null);

  useEffect(() => {
    getReadiness().then(setReady).catch(() => {});
    getModels().then(setModels).catch(() => setModels([]));
  }, []);

  return (
    <div>
      <h1 className="page-title">Settings & System Info</h1>
      <p className="page-subtitle">System configuration, environment status, and limitations.</p>

      <div className="card">
        <div className="card-title">API Connection</div>
        <table>
          <tbody>
            <tr><td className="text-muted">Backend URL</td><td><code>{import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'}</code></td></tr>
            <tr><td className="text-muted">API Docs</td><td><a href="http://127.0.0.1:8000/api/docs" target="_blank" rel="noreferrer" style={{ color: '#818cf8' }}>http://127.0.0.1:8000/api/docs</a></td></tr>
          </tbody>
        </table>
      </div>

      {models && (
        <div className="card">
          <div className="card-title">Registered Models</div>
          {models.length === 0 ? (
            <p className="text-sm text-muted">
              No model is registered. Videos are processed up to feature extraction and finish as{' '}
              <code>completed_without_prediction</code>; no prediction is ever invented.
            </p>
          ) : (
            <table>
              <thead><tr><th>Type</th><th>Version</th><th>Active</th><th>Video inference</th><th>Detail</th></tr></thead>
              <tbody>
                {models.map(m => (
                  <tr key={m.id}>
                    <td>{m.model_type}{m.synthetic ? ' (synthetic)' : ''}</td>
                    <td>{m.version}</td>
                    <td>{m.is_active ? 'yes' : 'no'}</td>
                    <td>
                      <span className={`badge ${m.usable_for_video_inference ? 'badge-green' : 'badge-red'}`}>
                        {m.usable_for_video_inference ? 'usable' : 'not usable'}
                      </span>
                    </td>
                    <td className="text-sm text-muted">{m.status_detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {ready && (
        <div className="card">
          <div className="card-title">Component Status</div>
          <table>
            <tbody>
              {ready.components.map(c => (
                <tr key={c.name}>
                  <td className="text-muted">{c.name}</td>
                  <td>
                    <span className={`badge ${c.status === 'ok' ? 'badge-green' : 'badge-red'}`}>{c.status}</span>
                  </td>
                  <td className="text-muted text-sm">{c.detail ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <div className="card-title">Model Configuration</div>
        <table>
          <tbody>
            <tr><td className="text-muted">CNN backbone</td><td>SimpleCNN (4 blocks, 256-dim output)</td></tr>
            <tr><td className="text-muted">LSTM</td><td>2-layer, 128 hidden dim, packed masking</td></tr>
            <tr><td className="text-muted">Fusion</td><td>64-dim output (visual-only or multimodal)</td></tr>
            <tr><td className="text-muted">QAOA</td><td>4 qubits, depth 2, COBYLA optimizer, PennyLane</td></tr>
            <tr><td className="text-muted">VQC</td><td>4 qubits, depth 3, CZ entanglement ring, backprop</td></tr>
            <tr><td className="text-muted">Quantum backend</td><td>default.qubit (CPU simulator)</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-title">Privacy & Data Retention</div>
        <table>
          <tbody>
            <tr><td className="text-muted">Raw video retention</td><td>Deleted once processing completes and its outputs are saved (RETAIN_RAW_VIDEO=false). Failed/cancelled jobs keep it for retry until deleted.</td></tr>
            <tr><td className="text-muted">Frame storage</td><td>Target crops, tracking record and features stored in data/frames/&lt;job_id&gt;/ until the job is deleted</td></tr>
            <tr><td className="text-muted">External transmission</td><td>None — fully local processing</td></tr>
            <tr><td className="text-muted">Facial identification</td><td>Not performed — operator selects target visually</td></tr>
            <tr><td className="text-muted">Age estimation</td><td>Not performed</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="card-title">Known Limitations</div>
        <ul style={{ paddingLeft: '1.5rem', lineHeight: 2, fontSize: '0.875rem', color: '#94a3b8' }}>
          <li>No real labelled dataset is included. All trained models use synthetic data only.</li>
          <li>Quantum circuits run on a CPU simulator — not physical quantum hardware.</li>
          <li>QAOA on 4 qubits is feasible; scaling to many features requires dimensionality reduction first.</li>
          <li>Federated learning is simulated on a single machine — not distributed across real clients.</li>
          <li>The VQC inference path requires a pipeline bundle (trained CNN/LSTM/fusion weights + QAOA mapping + compatible VQC). The QAOA+VQC experiment does not produce one, so no prediction is made until one is supplied.</li>
          <li>Screen-time data must be supplied separately — it cannot be inferred from video alone.</li>
          <li>Video tracking may fail in crowded scenes, poor lighting, or heavy occlusion.</li>
          <li>All predictions are experimental and NOT clinical diagnoses.</li>
        </ul>
      </div>

      <div className="card">
        <div className="card-title">Quick Commands</div>
        <p className="text-muted text-sm">Run from the project root (quantum_federated_behaviour_ai/):</p>
        <pre className="mt-1">{`# Start backend
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Start frontend (from frontend/)
npm run dev

# Run tests
python -m pytest tests/ -v

# Check environment
python scripts/check_environment.py`}</pre>
      </div>
    </div>
  );
}
