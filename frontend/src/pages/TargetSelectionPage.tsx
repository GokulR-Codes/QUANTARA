import { useCallback, useEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  apiErrorMessage, getFrameInfo, getJobStatus, previewFrameUrl, processVideo, selectTarget,
} from '../api/client';
import type { FrameInfo, JobResponse, TargetSelectionResponse } from '../types';
import type { Box, Point } from '../utils/bbox';
import { boxFromFields, boxFromPoints, clientToSource, sourceToDisplay, validateBox } from '../utils/bbox';

const SELECTABLE = new Set(['uploaded', 'failed', 'cancelled']);

type Fields = { x: string; y: string; w: string; h: string };
const toFields = (b: Box | null): Fields =>
  b ? { x: String(b.x), y: String(b.y), w: String(b.w), h: String(b.h) } : { x: '', y: '', w: '', h: '' };

export default function TargetSelectionPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<JobResponse | null>(null);
  const [frame, setFrame] = useState<FrameInfo | null>(null);
  const [imgError, setImgError] = useState<string | null>(null);
  const [timeMs, setTimeMs] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [fields, setFields] = useState<Fields>(toFields(null));
  const [confirmed, setConfirmed] = useState<TargetSelectionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [displayRect, setDisplayRect] = useState<{ left: number; top: number; width: number; height: number } | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragStart = useRef<Point | null>(null);

  const loadFrame = useCallback(async (tMs: number) => {
    if (!jobId) return;
    setError(null);
    setImgError(null);
    try {
      const info = await getFrameInfo(jobId, tMs);
      setFrame(info);
      setTimeMs(info.time_ms);
      setBox(null);
      setFields(toFields(null));
      setConfirmed(null);
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not load a preview frame'));
    }
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    (async () => {
      try {
        const j = await getJobStatus(jobId);
        if (cancelled) return;
        setJob(j);
        if (SELECTABLE.has(j.status)) await loadFrame(0);
      } catch (err) {
        if (!cancelled) setError(apiErrorMessage(err, 'Could not load the job'));
      }
    })();
    return () => { cancelled = true; };
  }, [jobId, loadFrame]);

  const measure = useCallback(() => {
    const r = imgRef.current?.getBoundingClientRect();
    if (r) setDisplayRect({ left: r.left, top: r.top, width: r.width, height: r.height });
  }, []);

  useEffect(() => {
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [measure]);

  const frameSize = frame ? { width: frame.width, height: frame.height } : null;
  const validation = validateBox(box, frameSize);

  const updateBox = (b: Box | null) => {
    setBox(b);
    setFields(toFields(b));
    setConfirmed(null);
    setError(null);
  };

  const pointFromEvent = (e: ReactPointerEvent<HTMLDivElement>): Point | null => {
    const r = imgRef.current?.getBoundingClientRect();
    if (!r || !frameSize) return null;
    return clientToSource(e.clientX, e.clientY, r, frameSize);
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    const p = pointFromEvent(e);
    if (!p || busy) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragStart.current = p;
    updateBox(null);
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const start = dragStart.current;
    const p = pointFromEvent(e);
    if (!start || !p || !frameSize) return;
    setBox(boxFromPoints(start, p, frameSize));
  };

  const onPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    const start = dragStart.current;
    dragStart.current = null;
    const p = pointFromEvent(e);
    if (start && p && frameSize) updateBox(boxFromPoints(start, p, frameSize));
  };

  const onFieldChange = (key: keyof Fields, value: string) => {
    const next = { ...fields, [key]: value };
    setFields(next);
    setConfirmed(null);
    setBox(boxFromFields(next));
  };

  const confirm = async () => {
    if (!jobId || !frame || !box || validation) return;
    setBusy(true);
    setError(null);
    try {
      const res = await selectTarget(jobId, box, { width: frame.width, height: frame.height, index: frame.frame_index });
      setConfirmed(res);
    } catch (err) {
      setConfirmed(null);
      setError(apiErrorMessage(err, 'The server rejected this selection'));
    } finally {
      setBusy(false);
    }
  };

  const start = async () => {
    if (!jobId || !confirmed) return;
    setBusy(true);
    setError(null);
    try {
      await processVideo(jobId);
      navigate(`/results/${jobId}`);
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not start processing'));
      setBusy(false);
    }
  };

  if (!jobId) return <div className="alert alert-error">No job specified.</div>;
  if (job && !SELECTABLE.has(job.status)) {
    return (
      <div>
        <h1 className="page-title">Select Participant</h1>
        <div className="alert alert-warning">
          This job is already <strong>{job.status}</strong>, so the participant can no longer be changed.{' '}
          <Link to={`/results/${jobId}`}>View its results</Link>.
        </div>
      </div>
    );
  }

  const overlay = box && displayRect && frameSize ? sourceToDisplay(box, displayRect, frameSize) : null;
  const durationMs = Math.max(0, Math.floor((job?.video_metadata?.duration_s ?? 0) * 1000));

  return (
    <div>
      <h1 className="page-title">Select the Participant</h1>
      <p className="page-subtitle">
        Draw a box around the one person to be analysed, then confirm. Nothing is processed until you do.
      </p>

      <div className="alert alert-info">
        The system follows <strong>only the region you select</strong>. It does not recognise or identify people, or
        estimate age. If the target is lost or becomes uncertain, those frames are skipped and reported — they are never
        replaced with another person. Make sure the box encloses the intended participant and nobody else.
      </div>

      {error && <div className="alert alert-error" role="alert">{error}</div>}

      <div className="card">
        <div className="card-title">
          Frame {frame ? `#${frame.frame_index} at ${(frame.time_ms / 1000).toFixed(2)} s` : '…'}
          {frame && <span className="text-muted"> — {frame.width}×{frame.height} px</span>}
        </div>

        {frame && (
          <div className="selector-stage">
            <img
              key={frame.frame_index}
              ref={imgRef}
              src={previewFrameUrl(jobId, frame.frame_index)}
              alt={`Video frame ${frame.frame_index}`}
              draggable={false}
              onLoad={() => {
                measure();
                const el = imgRef.current;
                if (el && (el.naturalWidth !== frame.width || el.naturalHeight !== frame.height)) {
                  setImgError(`Preview is ${el.naturalWidth}×${el.naturalHeight} but the video is ${frame.width}×${frame.height}; selection disabled.`);
                }
              }}
              onError={() => setImgError('The preview frame could not be loaded (the raw video may have been deleted).')}
            />
            {!imgError && (
              <div
                className="selector-overlay"
                role="application"
                aria-label="Draw a box around the participant"
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={() => { dragStart.current = null; }}
              >
                {overlay && (
                  <div
                    className={'selector-box' + (confirmed ? ' confirmed' : '')}
                    style={{ left: overlay.x, top: overlay.y, width: overlay.w, height: overlay.h }}
                  />
                )}
              </div>
            )}
          </div>
        )}
        {imgError && <div className="alert alert-error mt-2">{imgError}</div>}

        {durationMs > 0 && (
          <div className="selector-time">
            <label htmlFor="frame-time">Choose a different frame:</label>
            <input
              id="frame-time" type="range" min={0} max={durationMs} step={100} value={Math.min(timeMs, durationMs)}
              onChange={e => setTimeMs(Number(e.target.value))}
            />
            <span className="text-sm">{(timeMs / 1000).toFixed(1)} s</span>
            <button className="btn btn-secondary btn-sm" onClick={() => loadFrame(timeMs)} disabled={busy}>
              Show this frame
            </button>
          </div>
        )}
        <p className="text-sm text-muted">
          Tracking starts at the frame you select; earlier frames are not analysed. Pick a frame where the participant is
          clearly visible.
        </p>

        <div className="selector-fields" aria-label="Box coordinates in source pixels">
          {(['x', 'y', 'w', 'h'] as const).map(k => (
            <label key={k}>
              {k === 'x' ? 'X (left)' : k === 'y' ? 'Y (top)' : k === 'w' ? 'Width' : 'Height'}
              <input
                inputMode="numeric" value={fields[k]} onChange={e => onFieldChange(k, e.target.value)}
                aria-label={`Box ${k} in pixels`}
              />
            </label>
          ))}
        </div>
        {box && validation && <div className="alert alert-warning">{validation}</div>}

        <div className="camera-controls">
          <button className="btn btn-secondary" onClick={() => updateBox(null)} disabled={busy || !box}>Clear box</button>
          <button className="btn btn-primary" onClick={confirm} disabled={busy || !box || !!validation || !!imgError}>
            Confirm this participant
          </button>
          <button className="btn btn-success" onClick={start} disabled={busy || !confirmed}>
            ▶ Start processing
          </button>
        </div>

        {confirmed && (
          <div className="alert alert-success mt-2" role="status">
            Participant confirmed: box ({confirmed.bbox.x}, {confirmed.bbox.y}) {confirmed.bbox.w}×{confirmed.bbox.h} px on
            frame #{confirmed.frame_index} ({(confirmed.frame_time_ms / 1000).toFixed(2)} s). Checked by the server against the
            real {confirmed.frame_width}×{confirmed.frame_height} frame.
          </div>
        )}
      </div>
    </div>
  );
}
