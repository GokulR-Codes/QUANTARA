// API response types matching backend Pydantic schemas

export type JobStatus =
  | 'uploaded'
  | 'queued'
  | 'processing'
  | 'tracking'
  | 'feature_extraction'
  | 'inference'
  | 'completed'
  | 'completed_without_prediction'
  | 'failed'
  | 'cancelled';

export type ExperimentType = 'classical' | 'qaoa_vqc' | 'federated';
export type ExperimentStatus = 'created' | 'queued' | 'running' | 'completed' | 'failed';

export interface VideoMetadata {
  duration_s?: number;
  fps?: number;
  width?: number;
  height?: number;
  codec?: string;
  file_size_bytes?: number;
  original_filename?: string;
}

export interface FrameProcessingSummary {
  frames_decoded: number;
  frames_sampled: number;
  frames_accepted: number;
  frames_rejected: number;
  frames_used: number;
  padding_applied: number;
  truncation_applied: boolean;
  subsampled?: boolean;
}

export interface BoundingBox { x: number; y: number; w: number; h: number }

export type IntervalStatus = 'lost' | 'ambiguous' | 'out_of_frame';

export interface InvalidInterval {
  status: IntervalStatus;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  n_samples: number;
  reasons: string[];
}

export interface TargetTrackingSummary {
  tracking_enabled: boolean;
  initial_bbox?: BoundingBox;
  selection?: { frame_index: number; time_ms: number; bbox: BoundingBox };
  tracker_backend?: string;
  timestamp_source?: string;
  samples_total?: number;
  frames_tracked: number;
  frames_lost: number;
  frames_ambiguous?: number;
  frames_out_of_frame?: number;
  frames_invalid?: number;
  tracking_coverage_pct?: number;
  tracking_failures: number;
  ambiguity_events?: number;
  resumed_after_ambiguity?: number;
  first_loss_ms?: number | null;
  crops_generated?: number;
  intervals?: InvalidInterval[];
  truncated?: boolean;
  tracking_error?: string;
}

export interface TargetSelection {
  bbox: BoundingBox;
  frame_index: number;
  frame_time_ms: number;
  frame_width?: number;
  frame_height?: number;
  selected_at?: string;
}

export interface FrameInfo {
  job_id: string;
  frame_index: number;
  time_ms: number;
  width: number;
  height: number;
  duration_s?: number;
}

export interface TargetSelectionResponse {
  job_id: string;
  target_set: boolean;
  bbox: BoundingBox;
  frame_index: number;
  frame_time_ms: number;
  frame_width: number;
  frame_height: number;
}

export interface FeatureExtractionInfo {
  n_embeddings?: number;
  n_padded_masked?: number;
  n_crops_available?: number;
  subsampled?: boolean;
  cnn_embedding_dim?: number;
  lstm_dim?: number;
  fused_dim?: number;
  min_required?: number;
  weights_provenance?: { kind?: string; note?: string; [k: string]: unknown };
}

export interface PredictionResult {
  predicted_class: number;
  predicted_label: string;
  class_probabilities?: number[];
  synthetic_model?: boolean;
  disclaimer: string;
}

export interface JobResponse {
  job_id: string;
  recording_id?: string;
  status: JobStatus;
  created_at: string;
  updated_at?: string;
  completed_at?: string;
  video_metadata?: VideoMetadata;
  frame_processing_summary?: FrameProcessingSummary;
  target_tracking_summary?: TargetTrackingSummary;
  target_selection?: TargetSelection;
  feature_extraction_status?: string;
  feature_extraction?: FeatureExtractionInfo;
  model_status?: string;
  model_status_detail?: string;
  model_provenance?: Record<string, unknown>;
  active_model_id?: string;
  model_version?: string;
  feature_schema_version?: string;
  selected_feature_names?: string[];
  quantum_configuration?: Record<string, unknown>;
  prediction?: PredictionResult;
  experiment_metrics?: Record<string, unknown>;
  warnings?: string[];
  errors?: string;
  artifact_references?: Record<string, string>;
  processing_duration_s?: number;
}

export interface UploadResponse {
  job_id: string;
  status: JobStatus;
  message: string;
  original_filename: string;
  file_size_bytes: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
}

export interface ComponentStatus {
  name: string;
  status: string;
  detail?: string;
}

export interface ReadinessResponse {
  ready: boolean;
  api_operational: boolean;
  database_operational: boolean;
  model_available: boolean;
  quantum_available: boolean;
  components: ComponentStatus[];
  message: string;
}

export interface ModelInfo {
  id: string;
  model_type: string;
  version: string;
  is_active: boolean;
  metrics?: Record<string, unknown>;
  created_at: string;
  feature_schema_version?: string;
  selected_feature_names?: string[];
  usable_for_video_inference?: boolean;
  status_detail?: string;
  synthetic?: boolean | null;
}

export interface ExperimentResponse {
  experiment_id: string;
  name: string;
  experiment_type: ExperimentType;
  status: ExperimentStatus;
  config?: Record<string, unknown>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  results?: Record<string, unknown>;
  notes?: string[];
}

export interface DatasetStatusResponse {
  dataset_available: boolean;
  total_samples: number;
  class_distribution?: Record<string, number>;
  manifest_path?: string;
  message: string;
}

export interface FederatedStatusResponse {
  federated_experiments: Array<{
    id: string;
    name: string;
    status: string;
    created_at?: string;
  }>;
  note: string;
}
