import { Link, useParams } from 'react-router-dom';
import { useJob } from '../hooks/useJob';
import { apiErrorMessage, requestPrediction, cancelJob } from '../api/client';
import { useState } from 'react';
import type { JobStatus } from '../types';

const STATUS_LABELS: Record<JobStatus, string> = {
  uploaded: 'Uploaded',
  queued: 'Queued',
  processing: 'Processing frames',
  tracking: 'Tracking target',
  feature_extraction: 'Extracting features',
  inference: 'Running quantum inference',
  completed: 'Completed',
  completed_without_prediction: 'Completed (no prediction)',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const BADGE_CLASS: Partial<Record<JobStatus, string>> = {
  completed: 'badge-green',
  completed_without_prediction: 'badge-yellow',
  failed: 'badge-red',
  cancelled: 'badge-gray',
};

const fmtSeconds = (ms: number | null | undefined): string =>
  ms === null || ms === undefined ? '—' : `${(ms / 1000).toFixed(1)} s`;

function StatusBadge({ status }: { status: JobStatus }) {
  const cls = BADGE_CLASS[status] ?? 'badge-blue';
  return <span className={`badge ${cls}`}>{STATUS_LABELS[status]}</span>;
}

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { job, loading, error, startPolling } = useJob(jobId);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const handlePredict = async () => {
    if (!jobId) return;
    setPredicting(true);
    setPredictError(null);
    try {
      await requestPrediction(jobId);
      startPolling(jobId);
    } catch (err: unknown) {
      setPredictError(apiErrorMessage(err, 'Prediction request failed'));
    } finally {
      setPredicting(false);
    }
  };

  const handleCancel = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await cancelJob(jobId);
      startPolling(jobId);
    } catch {
      /* ignore */
    } finally {
      setCancelling(false);
    }
  };

  const isTerminal = job && ['completed', 'completed_without_prediction', 'failed', 'cancelled'].includes(job.status);

  return (
    <div>
      <h1 className="page-title">Processing Results</h1>
      <p className="page-subtitle">Job ID: <code>{jobId}</code></p>

      {error && <div className="alert alert-error">{error}</div>}
      {predictError && <div className="alert alert-error">{predictError}</div>}

      {loading && !job && (
        <div className="card"><span className="spin">⟳</span> Loading job status…</div>
      )}

      {job && (
        <>
          {/* Status */}
          <div className="card">
            <div className="flex items-center justify-between">
              <div>
                <div className="card-title">Status</div>
                <StatusBadge status={job.status} />
              </div>
              <div className="flex gap-2">
                {!isTerminal && (
                  <button className="btn btn-danger btn-sm" onClick={handleCancel} disabled={cancelling}>
                    {cancelling ? '…' : 'Cancel'}
                  </button>
                )}
                {(job.status === 'completed_without_prediction') && (
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={handlePredict}
                    disabled={predicting}
                  >
                    {predicting ? <><span className="spin">⟳</span> Running…</> : '⚛ Run Inference'}
                  </button>
                )}
              </div>
            </div>
            {job.errors && <div className="alert alert-error mt-2">{job.errors}</div>}
            {['uploaded', 'failed', 'cancelled'].includes(job.status) && (
              <div className="alert alert-info mt-2">
                {job.status === 'uploaded'
                  ? 'This video has not been processed yet. '
                  : 'You can retry this video. '}
                <Link to={`/select/${job.job_id}`}>
                  {job.target_selection ? 'Review or change the selected participant' : 'Select the participant'}
                </Link>{' '}
                to continue.
              </div>
            )}
            {job.warnings?.map((w, i) => (
              <div key={i} className="alert alert-warning mt-1">{w}</div>
            ))}
          </div>

          {/* Video Metadata */}
          {job.video_metadata && (
            <div className="card">
              <div className="card-title">Video Metadata</div>
              <table>
                <tbody>
                  {job.video_metadata.original_filename && (
                    <tr><td className="text-muted">File</td><td>{job.video_metadata.original_filename}</td></tr>
                  )}
                  {job.video_metadata.duration_s && (
                    <tr><td className="text-muted">Duration</td><td>{job.video_metadata.duration_s.toFixed(1)}s</td></tr>
                  )}
                  {job.video_metadata.fps && (
                    <tr><td className="text-muted">FPS</td><td>{job.video_metadata.fps.toFixed(1)}</td></tr>
                  )}
                  {job.video_metadata.width && (
                    <tr><td className="text-muted">Resolution</td><td>{job.video_metadata.width}×{job.video_metadata.height}</td></tr>
                  )}
                  {job.video_metadata.codec && (
                    <tr><td className="text-muted">Codec</td><td>{job.video_metadata.codec}</td></tr>
                  )}
                  {job.video_metadata.file_size_bytes && (
                    <tr><td className="text-muted">Size</td><td>{(job.video_metadata.file_size_bytes / 1024 / 1024).toFixed(2)} MB</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          )}

          {/* Frame Processing */}
          {job.frame_processing_summary && (
            <div className="card">
              <div className="card-title">Frame Processing</div>
              <div className="grid-3">
                {Object.entries(job.frame_processing_summary).map(([k, v]) => (
                  <div key={k}>
                    <div className="stat-value" style={{ fontSize: '1.4rem' }}>{String(v)}</div>
                    <div className="stat-label">{k.replace(/_/g, ' ')}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Participant selection */}
          {job.target_selection && (
            <div className="card">
              <div className="card-title">Selected Participant</div>
              <p className="text-sm">
                Box ({job.target_selection.bbox.x}, {job.target_selection.bbox.y}),{' '}
                {job.target_selection.bbox.w}×{job.target_selection.bbox.h} px on frame #{job.target_selection.frame_index}{' '}
                ({fmtSeconds(job.target_selection.frame_time_ms)}) of a {job.target_selection.frame_width}×
                {job.target_selection.frame_height} video — chosen by the operator.
              </p>
            </div>
          )}

          {/* Target Tracking */}
          {job.target_tracking_summary && job.target_tracking_summary.tracking_enabled && (
            <div className="card">
              <div className="card-title">Target Tracking</div>
              <table>
                <tbody>
                  <tr><td className="text-muted">Tracker</td><td>{job.target_tracking_summary.tracker_backend ?? '—'}</td></tr>
                  <tr><td className="text-muted">Sampled frames</td><td>{job.target_tracking_summary.samples_total ?? '—'}</td></tr>
                  <tr><td className="text-muted">Trusted (crop produced)</td><td>{job.target_tracking_summary.frames_tracked}</td></tr>
                  <tr><td className="text-muted">Ambiguous</td><td>{job.target_tracking_summary.frames_ambiguous ?? 0}</td></tr>
                  <tr><td className="text-muted">Lost / out of frame</td><td>{job.target_tracking_summary.frames_lost}</td></tr>
                  <tr><td className="text-muted">Coverage</td>
                    <td>{job.target_tracking_summary.tracking_coverage_pct?.toFixed(1) ?? '—'}%</td>
                  </tr>
                </tbody>
              </table>
              {(job.target_tracking_summary.intervals?.length ?? 0) > 0 && (
                <>
                  <p className="text-sm mt-2"><strong>Intervals where the target was not trusted</strong> (no crops were generated for these):</p>
                  <ul className="interval-list">
                    {job.target_tracking_summary.intervals!.map((iv, i) => (
                      <li key={i}>
                        <span className={`badge ${iv.status === 'ambiguous' ? 'badge-yellow' : 'badge-red'}`}>{iv.status}</span>{' '}
                        {fmtSeconds(iv.start_ms)} – {fmtSeconds(iv.end_ms)} ({iv.n_samples} sample{iv.n_samples === 1 ? '' : 's'})
                        {iv.reasons.length > 0 && <span className="text-muted"> — {iv.reasons.join('; ')}</span>}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              <p className="text-sm text-muted mt-2">
                The tracker follows the region you selected; it can drift onto a similar-looking person without noticing.
                Frames that fail the appearance/motion checks are excluded, but the checks are heuristics — review the
                tracking record if the result matters.
              </p>
            </div>
          )}

          {/* Feature extraction */}
          {job.feature_extraction && (
            <div className="card">
              <div className="card-title">Feature Extraction (CNN → LSTM → fusion)</div>
              <p className="text-sm">
                <strong>Status:</strong>{' '}
                <span className={`badge ${job.feature_extraction_status === 'completed' ? 'badge-green' : 'badge-yellow'}`}>
                  {job.feature_extraction_status}
                </span>
              </p>
              {job.feature_extraction.n_embeddings !== undefined && (
                <p className="text-sm mt-1">
                  {job.feature_extraction.n_embeddings} ordered CNN embeddings ({job.feature_extraction.cnn_embedding_dim}-d) →
                  LSTM ({job.feature_extraction.lstm_dim}-d) → fused vector ({job.feature_extraction.fused_dim}-d);{' '}
                  {job.feature_extraction.n_padded_masked} masked padding position(s).
                </p>
              )}
              {job.feature_extraction.weights_provenance?.kind === 'untrained_random_init' && (
                <div className="alert alert-warning mt-2">
                  These features came from <strong>untrained, randomly initialised</strong> network weights. They show that the
                  pipeline ran, but carry no learned meaning and cannot support a prediction.
                </div>
              )}
              {job.feature_extraction.min_required !== undefined && (
                <div className="alert alert-warning mt-2">
                  Only {job.feature_extraction.n_crops_available} trusted crop(s) were available; at least{' '}
                  {job.feature_extraction.min_required} are required, so no features were computed.
                </div>
              )}
            </div>
          )}

          {/* Model Status */}
          <div className="card">
            <div className="card-title">Model Status</div>
            <p className="text-sm">
              <strong>Status:</strong>{' '}
              <span className={`badge ${job.model_status === 'inference_completed' ? 'badge-green' : 'badge-yellow'}`}>
                {job.model_status ?? 'not_run'}
              </span>
            </p>
            {job.model_status_detail && (
              <p className="text-sm mt-1 text-muted">{job.model_status_detail}</p>
            )}
            {job.active_model_id && (
              <p className="text-sm mt-1"><strong>Model ID:</strong> {job.active_model_id}</p>
            )}
            {job.prediction?.synthetic_model && (
              <div className="alert alert-warning mt-2">
                This output came from a <strong>synthetic-trained test model</strong>. It demonstrates the software path only
                and has no behavioural meaning.
              </div>
            )}
            {job.selected_feature_names && (
              <p className="text-sm mt-1"><strong>QAOA-selected features:</strong> {job.selected_feature_names.join(', ')}</p>
            )}
          </div>

          {/* Prediction */}
          {job.prediction ? (
            <div className="card" style={{ border: '1px solid #059669' }}>
              <div className="card-title" style={{ color: '#6ee7b7' }}>⚛ Prediction Result</div>
              <div className="stat-value">{job.prediction.predicted_label}</div>
              <div className="stat-label">Predicted risk level (class {job.prediction.predicted_class})</div>
              {job.prediction.class_probabilities && (
                <div className="mt-2">
                  <p className="text-sm text-muted">Class probabilities:</p>
                  {job.prediction.class_probabilities.map((p, i) => (
                    <div key={i} className="flex items-center gap-2 mt-1">
                      <span className="text-sm" style={{ width: '80px' }}>Class {i}</span>
                      <div className="progress-bar-wrap w-full">
                        <div className="progress-bar" style={{ width: `${p * 100}%`, background: i === job.prediction!.predicted_class ? '#22c55e' : '#4f46e5' }} />
                      </div>
                      <span className="text-sm" style={{ width: '50px' }}>{(p * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
              <div className="alert alert-warning mt-2">
                {job.prediction.disclaimer}
              </div>
            </div>
          ) : isTerminal && !job.prediction && (
            <div className="card">
              <div className="card-title">Prediction</div>
              <p className="text-muted text-sm">
                No prediction was produced. Model status: <strong>{job.model_status ?? 'unknown'}</strong>.
              </p>
              {job.status === 'completed_without_prediction' && (
                <button
                  className="btn btn-primary btn-sm mt-2"
                  onClick={handlePredict}
                  disabled={predicting}
                >
                  {predicting ? <><span className="spin">⟳</span> Running…</> : '⚛ Run Inference Now'}
                </button>
              )}
            </div>
          )}

          {/* Duration */}
          {job.processing_duration_s && (
            <p className="text-muted text-sm mt-1">Processing time: {job.processing_duration_s.toFixed(2)}s</p>
          )}
        </>
      )}
    </div>
  );
}
