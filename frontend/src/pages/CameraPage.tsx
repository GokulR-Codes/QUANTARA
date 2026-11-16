import { useState, useRef, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useCamera } from '../hooks/useCamera';
import { apiErrorMessage, uploadVideo } from '../api/client';
import { getChild, createAssessment } from '../api/authClient';
import { useAuth } from '../contexts/AuthContext';
import type { ChildProfile } from '../types/auth';

function formatTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

export default function CameraPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { isAuthenticated } = useAuth();
  const childId = searchParams.get('child');

  const [selectedChild, setSelectedChild] = useState<ChildProfile | null>(null);
  const [childError, setChildError] = useState<string | null>(null);

  const {
    state, videoRef, previewUrl, elapsedSeconds, error,
    startCamera, stopCamera, startRecording, stopRecording, retake, getVideoFile,
  } = useCamera({ maxDurationSeconds: 300 });

  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const previewVideoRef = useRef<HTMLVideoElement | null>(null);

  // Load child profile when ?child=<id> is provided
  useEffect(() => {
    if (!childId || !isAuthenticated) return;
    getChild(childId)
      .then(setSelectedChild)
      .catch(() => setChildError('Could not load child profile. The session may have changed.'));
  }, [childId, isAuthenticated]);

  const handleSaveAndSelect = useCallback(async () => {
    const file = getVideoFile('camera_recording.webm');
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      const up = await uploadVideo(file, pct => setUploadProgress(pct));
      // If a child profile is selected, create an assessment session linking job + child
      if (selectedChild) {
        try {
          await createAssessment(selectedChild.id, up.job_id);
        } catch {
          // Assessment link failure is non-fatal — job still navigates correctly
        }
      }
      navigate(`/select/${up.job_id}`);          // operator selects participant
    } catch (err: unknown) {
      setUploadError(apiErrorMessage(err, 'Upload failed'));
    } finally {
      setUploading(false);
    }
  }, [getVideoFile, navigate, selectedChild]);

  const isIdle = state === 'idle';
  const isRequesting = state === 'requesting';
  const isActive = state === 'active';
  const isRecording = state === 'recording';
  const isPreview = state === 'preview';

  return (
    <div>
      <h1 className="page-title">Camera Recording</h1>
      <p className="page-subtitle">
        Record a video for processing. The camera will not start until you explicitly press "Start Camera".
      </p>

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

      <div className="alert alert-info">
        <strong>Privacy Notice:</strong> This application processes video locally.
        No video is sent externally. Recording requires your explicit consent each session.
        A visible indicator will show when recording is active.
      </div>
      {error && <div className="alert alert-error">{error}</div>}
      {uploadError && <div className="alert alert-error">{uploadError}</div>}

      {/* Camera preview / recorded preview */}
      <div className="camera-preview-wrapper">
        {/* Live camera feed */}
        <video
          ref={videoRef}
          className="camera-preview"
          style={{ display: isActive || isRecording ? 'block' : 'none' }}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
        />

        {/* Recorded video preview */}
        {isPreview && previewUrl && (
          <video
            ref={previewVideoRef}
            className="camera-preview"
            src={previewUrl}
            controls
            aria-label="Recorded video preview"
          />
        )}

        {/* Idle / error / denied overlay */}
        {(isIdle || isRequesting || state === 'permission_denied' || state === 'error') && (
          <div className="camera-overlay">
            {isIdle && <span style={{ fontSize: '3rem' }}>📷</span>}
            {isRequesting && <span className="spin" style={{ fontSize: '2rem' }}>⟳</span>}
            {state === 'permission_denied' && <span style={{ fontSize: '2rem' }}>🚫</span>}
            <p>{isIdle ? 'Camera not started' : isRequesting ? 'Requesting permission…' : (error ?? 'Camera unavailable')}</p>
          </div>
        )}

        {/* Recording indicator */}
        {isRecording && (
          <div className="recording-indicator" aria-live="polite">
            <span className="dot dot-red" />
            <span>REC</span>
            <span className="timer">{formatTime(elapsedSeconds)}</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="camera-controls">
        {isIdle && (
          <button className="btn btn-primary" onClick={startCamera} aria-label="Start camera">
            📷 Start Camera
          </button>
        )}

        {isActive && (
          <>
            <button className="btn btn-danger" onClick={startRecording} aria-label="Start recording">
              ⏺ Start Recording
            </button>
            <button className="btn btn-secondary" onClick={stopCamera}>
              ✕ Close Camera
            </button>
          </>
        )}

        {isRecording && (
          <button className="btn btn-danger" onClick={stopRecording} aria-label="Stop recording">
            ⏹ Stop Recording
          </button>
        )}

        {isPreview && !uploading && (
          <>
            <button className="btn btn-secondary" onClick={retake}>
              🔄 Retake
            </button>
            <button
              className="btn btn-success"
              onClick={handleSaveAndSelect}
              disabled={uploading}
            >
              ✓ Save & Select Participant
            </button>
          </>
        )}

        {uploading && (
          <div style={{ flex: 1 }}>
            <p className="text-sm text-muted">Uploading…</p>
            <div className="progress-bar-wrap">
              <div className="progress-bar" style={{ width: `${uploadProgress}%` }} />
            </div>
            <p className="text-sm text-muted">{uploadProgress}%</p>
          </div>
        )}
      </div>

      {/* Target selection note */}
      {isPreview && (
        <div className="card mt-2">
          <div className="card-title">Participant Selection</div>
          <p className="text-sm text-muted">
            After saving, you will be taken to a screen where you draw a box around the participant to analyse and confirm
            it. Nothing is processed until you do. The system tracks only the region you select — it does not
            identify people by appearance or estimate age.
          </p>
        </div>
      )}

      {/* Fallback link */}
      <div className="card mt-2">
        <div className="card-title">Alternatively</div>
        <p className="text-sm text-muted">If your camera is unavailable, you can upload a pre-recorded video:</p>
        <a href="/upload" className="btn btn-secondary btn-sm mt-1">Go to Upload Page</a>
      </div>
    </div>
  );
}
