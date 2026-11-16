// Q-CARE additional types — guardian auth, children, assessments

// ── Auth ──────────────────────────────────────────────────────────────────

export interface Guardian {
  id: string;
  email: string;
  username: string;
  full_name: string;
  mobile?: string;
  relationship_to_child?: string;
  country?: string;
  is_verified: boolean;
  role: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  guardian: Guardian;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  confirm_password: string;
  full_name: string;
  mobile?: string;
  relationship_to_child?: string;
  country?: string;
}

export interface LoginRequest {
  email_or_username: string;
  password: string;
}

// ── Age groups ────────────────────────────────────────────────────────────

export type AgeGroup = '5_7' | '7_10' | '10_12';

export const AGE_GROUP_LABELS: Record<AgeGroup, string> = {
  '5_7': '5–7 Years (Group A)',
  '7_10': '7–10 Years (Group B)',
  '10_12': '10–12 Years (Group C)',
};

export const AGE_GROUPS: AgeGroup[] = ['5_7', '7_10', '10_12'];

// ── Children ──────────────────────────────────────────────────────────────

export interface ChildProfile {
  id: string;
  guardian_id: string;
  display_name: string;
  age_group: AgeGroup;
  age_group_label: string;
  date_of_birth?: string;
  age_verified_status: string;
  gender?: string;
  notes?: string;
  is_active: boolean;
  guardian_consent_given: boolean;
  consent_given_at?: string;
  created_at: string;
  updated_at: string;
  assessment_count: number;
}

export interface CreateChildRequest {
  display_name: string;
  age_group: AgeGroup;
  date_of_birth?: string;
  gender?: string;
  notes?: string;
  guardian_consent_given: boolean;
  consent_video_assessment: boolean;
  consent_retain_video: boolean;
  consent_research_training: boolean;
  consent_biometric_research: boolean;
}

// ── Assessments ───────────────────────────────────────────────────────────

export interface AssessmentSession {
  id: string;
  guardian_id: string;
  child_id: string;
  child_name?: string;
  job_id?: string;
  age_group: AgeGroup;
  age_group_label: string;
  age_at_assessment?: number;
  status: string;
  model_version?: string;
  result_summary?: Record<string, unknown>;
  created_at: string;
  completed_at?: string;
  notes?: string;
}

// ── Age model status ───────────────────────────────────────────────────────

export interface AgeGroupModelStatus {
  age_group: AgeGroup;
  label: string;
  has_active_model: boolean;
  model_id?: string;
  model_version?: string;
  is_approved: boolean;
  validation_metrics?: Record<string, unknown>;
  processing_config_summary: Record<string, unknown>;
  research_notes: string;
}
