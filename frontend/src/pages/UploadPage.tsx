import { useState, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { apiErrorMessage, uploadVideo } from '../api/client';
import { getChild, createAssessment } from '../api/authClient';
import { useAuth } from '../contexts/AuthContext';
import type { ChildProfile } from '../types/auth';

export default function UploadPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { isAuthenticated } = useAuth();
  const childId = searchParams.get('child');

  const [selectedChild, setSelectedChild] = useState<ChildProfile | null>(null);
  const [childError, setChildError] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const ACCEPTED = '.mp4,.avi,.mov,.mkv,.webm';
  const MAX_SIZE_MB = 500;

  // Load child profile when ?child=<id> is provided
  useEffect(() => {
    if (!childId || !isAuthenticated) return;
    getChild(childId)
      .then(setSelectedChild)
      .catch(() => setChildError('Could not load child profile.'));
  }, [childId, isAuthenticated]);

  const handleFile = (f: File) => {
    const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext)) {
      setError(`File format .${ext} is not accepted. Use: ${ACCEPTED}`);
      return;
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File is too large (${(f.size / 1024 / 1024).toFixed(0)} MB). Max: ${MAX_SIZE_MB} MB.`);
      return;
    }
    setFile(f);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  const handleSubmit = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const up = await uploadVideo(file, pct => setProgress(pct));
      // If a child profile is selected, link job to assessment session
      if (selectedChild) {
        try {
          await createAssessment(selectedChild.id, up.job_id);
        } catch {
          // non-fatal
        }
      }
      navigate(`/select/${up.job_id}`);          // operator selects participant
    } catch (err: unknown) {
      setError(apiErrorMessage(err, 'Upload failed'));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Upload Video</h1>
      <p className="page-subtitle">Upload a pre-recorded video for processing. Accepted: {ACCEPTED}. Max {MAX_SIZE_MB} MB.</p>

      {/* Child profile banner */}
      {selectedChild && (
        <div className="alert alert-info" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '1.5rem' }}>👶</span>
          <div>
            <strong>Assessment for: {selectedChild.display_name}</strong>
            <span className="badge badge-blue" style={{ marginLeft: '0.5rem' }}>
              {selectedChild.age_group_label}
            </span>
            <div className="text-sm text-muted mt-1">
              Age-specific model ({selectedChild.age_group}) will be used for inference.
            </div>
          </div>
        </div>
      )}
      {childError && <div className="alert alert-warning">{childError}</div>}
      {!selectedChild && !childError && (
        <div className="alert alert-warning">
          <strong>No child profile selected.</strong>{' '}
          Assessment will not be linked to a profile. To link, start from the{' '}
          <a href="/guardian">guardian dashboard</a> and select a child.
        </div>
      )}

      {error && <div className="alert alert-error">{error}</div>}

      <div
        className={`dropzone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input')?.click()}
        role="button"
        tabIndex={0}
        aria-label="Drop video file here or click to browse"
        onKeyDown={e => e.key === 'Enter' && document.getElementById('file-input')?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept={ACCEPTED}
          onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])}
          aria-label="Select video file"
        />
        <span style={{ fontSize: '2.5rem' }}>📁</span>
        <p className="mt-2">Drop your video here, or click to browse</p>
        <p className="text-muted text-sm mt-1">Supported: MP4, AVI, MOV, MKV, WebM</p>
      </div>

      {file && (
        <div className="card mt-2">
          <div className="card-title">Selected File</div>
          <table>
            <tbody>
              <tr><td className="text-muted">Name</td><td>{file.name}</td></tr>
              <tr><td className="text-muted">Size</td><td>{(file.size / 1024 / 1024).toFixed(2)} MB</td></tr>
              <tr><td className="text-muted">Type</td><td>{file.type || 'unknown'}</td></tr>
            </tbody>
          </table>
        </div>
      )}

      {uploading && (
        <div className="mt-2">
          <p className="text-sm text-muted">Uploading… {progress}%</p>
          <div className="progress-bar-wrap">
            <div className="progress-bar" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <div className="flex gap-2 mt-2">
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
          disabled={!file || uploading}
        >
          {uploading ? <><span className="spin">⟳</span> Uploading…</> : '⬆ Upload & Select Participant'}
        </button>
        {file && !uploading && (
          <button className="btn btn-secondary" onClick={() => setFile(null)}>Clear</button>
        )}
      </div>

      <div className="alert alert-info mt-2">
        After uploading you will be asked to select the participant to analyse. Processing starts only after
        you confirm that selection; you can then track progress and retrieve the full result.
      </div>
    </div>
  );
}
