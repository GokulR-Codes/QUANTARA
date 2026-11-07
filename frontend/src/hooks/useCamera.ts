// useCamera hook — full camera lifecycle with permission handling

import { useState, useRef, useCallback, useEffect } from 'react';

export type CameraState =
  | 'idle'
  | 'requesting'
  | 'active'
  | 'recording'
  | 'preview'
  | 'error'
  | 'permission_denied';

interface UseCameraOptions {
  maxDurationSeconds?: number;
}

interface UseCameraReturn {
  state: CameraState;
  stream: MediaStream | null;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  previewUrl: string | null;
  recordedBlob: Blob | null;
  elapsedSeconds: number;
  error: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  startRecording: () => void;
  stopRecording: () => void;
  retake: () => void;
  getVideoFile: (filename?: string) => File | null;
}

export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const { maxDurationSeconds = 600 } = options;

  const [state, setState] = useState<CameraState>('idle');
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const maxTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const _clearTimers = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (maxTimerRef.current) { clearTimeout(maxTimerRef.current); maxTimerRef.current = null; }
  };

  const stopCamera = useCallback(() => {
    _clearTimers();
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    setStream(null);
    if (videoRef.current) videoRef.current.srcObject = null;
    setState('idle');
  }, []);

  // Cleanup on unmount
  useEffect(() => () => {
    _clearTimers();
    streamRef.current?.getTracks().forEach(t => t.stop());
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera API is not supported in this browser. Try Chrome or Firefox over HTTPS/localhost.');
      setState('error');
      return;
    }
    setState('requesting');
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true,
      });
      streamRef.current = s;
      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        videoRef.current.muted = true;
        await videoRef.current.play();
      }
      setState('active');
    } catch (err) {
      const e = err as DOMException;
      if (e.name === 'NotAllowedError' || e.name === 'PermissionDeniedError') {
        setState('permission_denied');
        setError('Camera permission denied. Please allow camera access and try again.');
      } else if (e.name === 'NotFoundError') {
        setState('error');
        setError('No camera found. Please connect a webcam and try again.');
      } else if (e.name === 'NotReadableError') {
        setState('error');
        setError('Camera is in use by another application. Close it and try again.');
      } else {
        setState('error');
        setError(`Camera error: ${e.message || e.name}`);
      }
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!streamRef.current || state !== 'active') return;

    // Choose a supported MIME type
    const mimeType = ['video/webm;codecs=vp9,opus', 'video/webm', 'video/mp4']
      .find(t => MediaRecorder.isTypeSupported(t)) || '';

    chunksRef.current = [];
    const mr = new MediaRecorder(streamRef.current, mimeType ? { mimeType } : {});
    mediaRecorderRef.current = mr;

    mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    mr.onstop = () => {
      _clearTimers();
      const blob = new Blob(chunksRef.current, { type: mimeType || 'video/webm' });
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
      setRecordedBlob(blob);
      setState('preview');
    };

    mr.start(1000); // collect 1-second chunks
    setState('recording');
    setElapsedSeconds(0);

    timerRef.current = setInterval(() => setElapsedSeconds(s => s + 1), 1000);
    maxTimerRef.current = setTimeout(() => {
      if (mediaRecorderRef.current?.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    }, maxDurationSeconds * 1000);
  }, [state, maxDurationSeconds]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    _clearTimers();
  }, []);

  const retake = useCallback(() => {
    if (previewUrl) { URL.revokeObjectURL(previewUrl); setPreviewUrl(null); }
    setRecordedBlob(null);
    setElapsedSeconds(0);
    setState('active');
  }, [previewUrl]);

  const getVideoFile = useCallback((filename = 'recording.webm'): File | null => {
    if (!recordedBlob) return null;
    return new File([recordedBlob], filename, { type: recordedBlob.type });
  }, [recordedBlob]);

  return {
    state, stream, videoRef, previewUrl, recordedBlob,
    elapsedSeconds, error,
    startCamera, stopCamera, startRecording, stopRecording, retake, getVideoFile,
  };
}
