// Axios-based API client — all requests go to the FastAPI backend

import axios from 'axios';
import type {
  JobResponse,
  UploadResponse,
  HealthResponse,
  ReadinessResponse,
  ModelInfo,
  ExperimentResponse,
  DatasetStatusResponse,
  FederatedStatusResponse,
  FrameInfo,
  TargetSelectionResponse,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// ── Health ────────────────────────────────────────────────────────────────
export const getHealth = (): Promise<HealthResponse> =>
  api.get('/api/v1/health').then(r => r.data);

export const getReadiness = (): Promise<ReadinessResponse> =>
  api.get('/api/v1/ready').then(r => r.data);

// ── Models ────────────────────────────────────────────────────────────────
export const getModels = (): Promise<ModelInfo[]> =>
  api.get('/api/v1/models').then(r => r.data);

// ── Videos ───────────────────────────────────────────────────────────────
export const uploadVideo = (file: File, onProgress?: (pct: number) => void): Promise<UploadResponse> => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/api/v1/videos/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total));
    },
  }).then(r => r.data);
};

/** Resolve a time to the exact source frame the operator will see and select on. */
export const getFrameInfo = (jobId: string, tMs = 0): Promise<FrameInfo> =>
  api.get(`/api/v1/videos/${jobId}/frame-info`, { params: { t_ms: tMs } }).then(r => r.data);

/** URL of the full-resolution JPEG of one source frame (coordinates are source pixels). */
export const previewFrameUrl = (jobId: string, frameIndex: number): string =>
  `${BASE_URL}/api/v1/videos/${jobId}/preview-frame?frame_index=${frameIndex}`;

/** Submit the operator-drawn participant box. The server validates it against the real frame. */
export const selectTarget = (
  jobId: string,
  bbox: { x: number; y: number; w: number; h: number },
  frame: { width: number; height: number; index: number },
): Promise<TargetSelectionResponse> =>
  api.post(`/api/v1/videos/${jobId}/select-target`, {
    job_id: jobId,
    bbox_x: bbox.x,
    bbox_y: bbox.y,
    bbox_w: bbox.w,
    bbox_h: bbox.h,
    frame_width: frame.width,
    frame_height: frame.height,
    frame_index: frame.index,
  }).then(r => r.data);

/** Extract the server's validation message from an axios error. */
export const apiErrorMessage = (err: unknown, fallback: string): string => {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map(d => (d as { msg?: string }).msg ?? '').filter(Boolean).join('; ') || fallback;
  return err instanceof Error ? err.message : fallback;
};

export const processVideo = (jobId: string): Promise<{ job_id: string; status: string; message: string }> =>
  api.post(`/api/v1/videos/${jobId}/process`).then(r => r.data);

export const getJobStatus = (jobId: string): Promise<JobResponse> =>
  api.get(`/api/v1/videos/${jobId}/status`).then(r => r.data);

export const getJobResult = (jobId: string): Promise<JobResponse> =>
  api.get(`/api/v1/videos/${jobId}/result`).then(r => r.data);

export const cancelJob = (jobId: string): Promise<unknown> =>
  api.post(`/api/v1/videos/${jobId}/cancel`).then(r => r.data);

export const deleteJob = (jobId: string): Promise<unknown> =>
  api.delete(`/api/v1/videos/${jobId}`).then(r => r.data);

// ── Predict ──────────────────────────────────────────────────────────────
export const requestPrediction = (jobId: string, modelId?: string): Promise<JobResponse> =>
  api.post('/api/v1/predict', { job_id: jobId, model_id: modelId }).then(r => r.data);

// ── Experiments ───────────────────────────────────────────────────────────
export const createExperiment = (name: string, type: string, config?: Record<string, unknown>): Promise<ExperimentResponse> =>
  api.post('/api/v1/experiments', { name, experiment_type: type, config }).then(r => r.data);

export const listExperiments = (): Promise<ExperimentResponse[]> =>
  api.get('/api/v1/experiments').then(r => r.data);

export const getExperimentStatus = (id: string): Promise<ExperimentResponse> =>
  api.get(`/api/v1/experiments/${id}/status`).then(r => r.data);

export const getExperimentResults = (id: string): Promise<ExperimentResponse> =>
  api.get(`/api/v1/experiments/${id}/results`).then(r => r.data);

// ── Datasets ──────────────────────────────────────────────────────────────
export const getDatasetStatus = (): Promise<DatasetStatusResponse> =>
  api.get('/api/v1/datasets/status').then(r => r.data);

export const validateDataset = (): Promise<unknown> =>
  api.post('/api/v1/datasets/validate').then(r => r.data);

// ── Federated ─────────────────────────────────────────────────────────────
export const getFederatedStatus = (): Promise<FederatedStatusResponse> =>
  api.get('/api/v1/federated/status').then(r => r.data);

export default api;
