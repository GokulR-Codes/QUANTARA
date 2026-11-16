// Q-CARE Auth API client functions
import api from './client';
import type {
  AuthResponse, Guardian, RegisterRequest, LoginRequest,
  ChildProfile, CreateChildRequest, AssessmentSession, AgeGroupModelStatus,
} from '../types/auth';

const TOKEN_KEY = 'qcare_token';
const GUARDIAN_KEY = 'qcare_guardian';

// ── Token management ────────────────────────────────────────────────────

export const saveSession = (token: string, guardian: Guardian): void => {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(GUARDIAN_KEY, JSON.stringify(guardian));
  api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
};

export const loadSession = (): { token: string | null; guardian: Guardian | null } => {
  const token = localStorage.getItem(TOKEN_KEY);
  const guardianStr = localStorage.getItem(GUARDIAN_KEY);
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }
  return {
    token,
    guardian: guardianStr ? (JSON.parse(guardianStr) as Guardian) : null,
  };
};

export const clearSession = (): void => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(GUARDIAN_KEY);
  delete api.defaults.headers.common['Authorization'];
};

// ── Auth endpoints ───────────────────────────────────────────────────────

export const register = (req: RegisterRequest): Promise<{ message: string; guardian_id: string; email: string }> =>
  api.post('/api/v1/auth/register', req).then(r => r.data);

export const login = (req: LoginRequest): Promise<AuthResponse> =>
  api.post('/api/v1/auth/login', req).then(r => r.data);

export const logout = (): Promise<unknown> =>
  api.post('/api/v1/auth/logout').then(r => r.data);

export const verifyOtp = (otpCode: string, purpose = 'email_verification'): Promise<{ message: string }> =>
  api.post('/api/v1/auth/verify-otp', { otp_code: otpCode, purpose }).then(r => r.data);

export const resendOtp = (purpose = 'email_verification'): Promise<{ message: string }> =>
  api.post('/api/v1/auth/resend-otp', { purpose }).then(r => r.data);

export const forgotPassword = (email: string): Promise<{ message: string }> =>
  api.post('/api/v1/auth/forgot-password', { email }).then(r => r.data);

export const resetPassword = (
  email: string, otpCode: string, newPassword: string, confirmPassword: string,
): Promise<{ message: string }> =>
  api.post('/api/v1/auth/reset-password', {
    email, otp_code: otpCode, new_password: newPassword, confirm_password: confirmPassword,
  }).then(r => r.data);

export const getMe = (): Promise<Guardian> =>
  api.get('/api/v1/auth/me').then(r => r.data);

// ── Children endpoints ───────────────────────────────────────────────────

export const createChild = (req: CreateChildRequest): Promise<ChildProfile> =>
  api.post('/api/v1/children', req).then(r => r.data);

export const listChildren = (includeInactive = false): Promise<ChildProfile[]> =>
  api.get('/api/v1/children', { params: { include_inactive: includeInactive } }).then(r => r.data);

export const getChild = (childId: string): Promise<ChildProfile> =>
  api.get(`/api/v1/children/${childId}`).then(r => r.data);

export const updateChild = (childId: string, data: Partial<CreateChildRequest>): Promise<ChildProfile> =>
  api.put(`/api/v1/children/${childId}`, data).then(r => r.data);

export const deleteChild = (childId: string, permanent = false): Promise<{ message: string }> =>
  api.delete(`/api/v1/children/${childId}`, { params: { permanent } }).then(r => r.data);

export const listChildAssessments = (childId: string): Promise<AssessmentSession[]> =>
  api.get(`/api/v1/children/${childId}/assessments`).then(r => r.data);

// ── Assessment endpoints ─────────────────────────────────────────────────

export const createAssessment = (
  childId: string, jobId?: string, notes?: string,
): Promise<AssessmentSession> =>
  api.post('/api/v1/assessments', { child_id: childId, job_id: jobId, notes }).then(r => r.data);

export const listAssessments = (childId?: string, ageGroup?: string): Promise<AssessmentSession[]> =>
  api.get('/api/v1/assessments', { params: { child_id: childId, age_group: ageGroup } }).then(r => r.data);

/** Find the assessment session linked to a specific job_id (returns undefined if none). */
export const findAssessmentByJob = async (jobId: string): Promise<AssessmentSession | undefined> => {
  const all: AssessmentSession[] = await api.get('/api/v1/assessments').then(r => r.data);
  return all.find((a) => a.job_id === jobId);
};

export const getAssessment = (assessmentId: string): Promise<AssessmentSession> =>
  api.get(`/api/v1/assessments/${assessmentId}`).then(r => r.data);

export const deleteAssessment = (assessmentId: string): Promise<{ message: string }> =>
  api.delete(`/api/v1/assessments/${assessmentId}`).then(r => r.data);

// ── Age model status ─────────────────────────────────────────────────────

export const getAgeGroupStatuses = (): Promise<AgeGroupModelStatus[]> =>
  api.get('/api/v1/models/age-groups').then(r => r.data);

export const getAgeGroupStatus = (ageGroup: string): Promise<AgeGroupModelStatus> =>
  api.get(`/api/v1/models/age-groups/${ageGroup}`).then(r => r.data);
