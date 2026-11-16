// Child Profile Registration and Management page
import { useState, useEffect, type FormEvent } from 'react';
import { useNavigate, useParams, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
  createChild, getChild, updateChild, deleteChild, listChildAssessments,
} from '../api/authClient';
import { apiErrorMessage } from '../api/client';
import type { ChildProfile, AgeGroup, AssessmentSession } from '../types/auth';
import { AGE_GROUP_LABELS, AGE_GROUPS } from '../types/auth';

const inputStyle: React.CSSProperties = {
  width: '100%', padding: '10px 12px', borderRadius: 6,
  border: '1px solid #444', background: '#2a2a3e', color: '#fff',
  fontSize: '0.95rem', boxSizing: 'border-box', marginTop: 4,
};
const btnPrimary: React.CSSProperties = {
  padding: '10px 20px', borderRadius: 6, border: 'none',
  background: '#7c3aed', color: '#fff', fontSize: '0.95rem',
  cursor: 'pointer', fontWeight: 600,
};
const errStyle: React.CSSProperties = {
  background: '#3b0000', color: '#fca5a5', padding: '10px 12px',
  borderRadius: 6, marginTop: 8, fontSize: '0.9rem',
};
const successStyle: React.CSSProperties = { ...errStyle, background: '#004d26', color: '#86efac' };

function FormGroup({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ fontSize: '0.88rem', color: '#a0a0c0' }}>{label}</label>
      {children}
      {hint && <div style={{ fontSize: '0.78rem', color: '#666', marginTop: 3 }}>{hint}</div>}
    </div>
  );
}

// ── CREATE CHILD FORM ────────────────────────────────────────────────────

export function CreateChildPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    display_name: '', age_group: '' as AgeGroup | '', date_of_birth: '',
    gender: '', notes: '', guardian_consent_given: false,
    consent_video_assessment: false, consent_retain_video: false,
    consent_research_training: false, consent_biometric_research: false,
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault(); setError('');
    if (!form.age_group) { setError('Age category is required.'); return; }
    if (!form.guardian_consent_given) { setError('Guardian consent is required to create a child profile.'); return; }
    if (!form.consent_video_assessment) { setError('Consent for video assessment is required to use Q-CARE.'); return; }
    setLoading(true);
    try {
      const child = await createChild({
        display_name: form.display_name,
        age_group: form.age_group,
        date_of_birth: form.date_of_birth || undefined,
        gender: form.gender || undefined,
        notes: form.notes || undefined,
        guardian_consent_given: form.guardian_consent_given,
        consent_video_assessment: form.consent_video_assessment,
        consent_retain_video: form.consent_retain_video,
        consent_research_training: form.consent_research_training,
        consent_biometric_research: form.consent_biometric_research,
      });
      navigate(`/children/${child.id}`);
    } catch (err) { setError(apiErrorMessage(err, 'Failed to create profile.')); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ maxWidth: 640, margin: '2rem auto', padding: '0 1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <button onClick={() => navigate('/guardian')}
          style={{ background: 'none', border: 'none', color: '#7c3aed', cursor: 'pointer', fontSize: '1.1rem' }}>← Back</button>
        <h2 style={{ margin: 0, color: '#a78bfa' }}>Register Child Profile</h2>
      </div>

      <div style={{ background: '#1a1a0e', borderRadius: 8, padding: '12px 16px', borderLeft: '3px solid #d97706', marginBottom: 20 }}>
        <p style={{ margin: 0, color: '#fbbf24', fontSize: '0.85rem' }}>
          This profile must be created by the parent or legal guardian. The child must not independently create a profile or initiate an assessment.
        </p>
      </div>

      <form onSubmit={submit} style={{ background: '#1e1e2e', padding: '1.5rem', borderRadius: 10 }}>
        <FormGroup label="Child Display Name or Pseudonym *" hint="Use a nickname or pseudonym — do not use full legal name.">
          <input style={inputStyle} type="text" value={form.display_name}
            onChange={e => setForm(f => ({ ...f, display_name: e.target.value }))}
            required placeholder="e.g. Alex or Child 1" maxLength={100} />
        </FormGroup>

        {/* AGE CATEGORY — MANDATORY */}
        <FormGroup label="Age Category * (MANDATORY)" hint="Select the category that matches the child's current age. This determines which model configuration is used.">
          <select
            style={{ ...inputStyle, appearance: 'auto' }}
            value={form.age_group}
            onChange={e => setForm(f => ({ ...f, age_group: e.target.value as AgeGroup }))}
            required
          >
            <option value="">— Select age category —</option>
            {AGE_GROUPS.map(g => (
              <option key={g} value={g}>{AGE_GROUP_LABELS[g]}</option>
            ))}
          </select>
        </FormGroup>

        <FormGroup label="Date of Birth (optional)" hint="If provided, used to verify the age category. Not required.">
          <input style={inputStyle} type="date" value={form.date_of_birth}
            onChange={e => setForm(f => ({ ...f, date_of_birth: e.target.value }))} />
        </FormGroup>

        <FormGroup label="Gender (optional)" hint="Only provide if relevant to the research protocol.">
          <select style={{ ...inputStyle, appearance: 'auto' }}
            value={form.gender} onChange={e => setForm(f => ({ ...f, gender: e.target.value }))}>
            <option value="">Prefer not to say / Not specified</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </FormGroup>

        <FormGroup label="Notes (optional)">
          <textarea style={{ ...inputStyle, height: 60, resize: 'vertical' } as React.CSSProperties}
            value={form.notes} onChange={e => setForm(f => ({ ...f, notes: e.target.value }))}
            placeholder="Any relevant notes about this child profile" />
        </FormGroup>

        {/* CONSENT */}
        <div style={{ background: '#13131f', borderRadius: 8, padding: '1rem', marginBottom: 16 }}>
          <h4 style={{ margin: '0 0 12px', color: '#c4b5fd' }}>Guardian Consent</h4>
          <p style={{ color: '#888', fontSize: '0.82rem', marginTop: 0 }}>
            Please read the Q-CARE <Link to="/privacy" style={{ color: '#a78bfa' }}>Privacy & Consent</Link> policy before proceeding.
            Consent for different purposes must be given separately.
          </p>
          {[
            { key: 'guardian_consent_given', label: 'I confirm I am the parent/guardian and authorize this profile *', required: true },
            { key: 'consent_video_assessment', label: 'I consent to video recording and AI-based assessment for this child *', required: true },
            { key: 'consent_retain_video', label: 'I consent to retaining raw video files after processing (optional)', required: false },
            { key: 'consent_research_training', label: 'I consent to de-identified data being used for model research training (optional)', required: false },
            { key: 'consent_biometric_research', label: 'I consent to optional biometric research features (disabled by default, requires separate setup)', required: false },
          ].map(({ key, label, required }) => (
            <label key={key} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 10, cursor: 'pointer', fontSize: '0.88rem', color: '#c0c0d0' }}>
              <input type="checkbox" checked={form[key as keyof typeof form] as boolean}
                onChange={e => setForm(f => ({ ...f, [key]: e.target.checked }))}
                style={{ marginTop: 2, flexShrink: 0 }} />
              <span>{label}</span>
            </label>
          ))}
        </div>

        {error && <div style={errStyle}>{error}</div>}
        <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
          <button type="submit" style={btnPrimary} disabled={loading}>
            {loading ? 'Creating…' : 'Create Child Profile'}
          </button>
          <button type="button" onClick={() => navigate('/guardian')}
            style={{ ...btnPrimary, background: '#2a2a3e', border: '1px solid #444', color: '#a0a0c0' }}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
}

// ── CHILD PROFILE VIEW ───────────────────────────────────────────────────

export function ChildProfilePage() {
  const { childId } = useParams<{ childId: string }>();
  const navigate = useNavigate();
  const [child, setChild] = useState<ChildProfile | null>(null);
  const [assessments, setAssessments] = useState<AssessmentSession[]>([]);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');

  useEffect(() => {
    if (!childId) return;
    Promise.all([
      getChild(childId),
      listChildAssessments(childId),
    ]).then(([c, a]) => { setChild(c); setAssessments(a); }).catch(() => setError('Failed to load.'));
  }, [childId]);

  const handleDeactivate = async () => {
    if (!child || !confirm(`Deactivate profile for ${child.display_name}?`)) return;
    try {
      await deleteChild(child.id, false);
      setMsg('Profile deactivated.'); setChild(c => c ? { ...c, is_active: false } : c);
    } catch (err) { setError(apiErrorMessage(err, 'Failed.')); }
  };

  const handleDelete = async () => {
    if (!child || !confirm(`PERMANENTLY DELETE all records for ${child.display_name}? This cannot be undone.`)) return;
    setDeleting(true);
    try {
      await deleteChild(child.id, true);
      navigate('/guardian');
    } catch (err) { setError(apiErrorMessage(err, 'Delete failed.')); setDeleting(false); }
  };

  if (!child) return <div style={{ padding: '2rem', color: '#888' }}>{error || 'Loading…'}</div>;

  return (
    <div style={{ maxWidth: 800, margin: '2rem auto', padding: '0 1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <button onClick={() => navigate('/guardian')}
          style={{ background: 'none', border: 'none', color: '#7c3aed', cursor: 'pointer', fontSize: '1.1rem' }}>← Dashboard</button>
        <h2 style={{ margin: 0, color: '#a78bfa' }}>👶 {child.display_name}</h2>
        {!child.is_active && <span style={{ color: '#fca5a5', fontSize: '0.85rem' }}>(deactivated)</span>}
      </div>

      {error && <div style={errStyle}>{error}</div>}
      {msg && <div style={successStyle}>{msg}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        <div style={{ background: '#1e1e2e', borderRadius: 10, padding: '1.25rem' }}>
          <h3 style={{ margin: '0 0 12px', color: '#c4b5fd', fontSize: '1rem' }}>Profile Details</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
            <tbody>
              {[
                ['Display Name', child.display_name],
                ['Age Category', child.age_group_label],
                ['Age Verification', child.age_verified_status.replace(/_/g, ' ')],
                ['Date of Birth', child.date_of_birth ?? 'Not provided'],
                ['Gender', child.gender ?? 'Not specified'],
                ['Consent Status', child.guardian_consent_given ? '✅ Consented' : '❌ No consent'],
                ['Assessments', String(child.assessment_count)],
                ['Status', child.is_active ? '✅ Active' : '❌ Deactivated'],
                ['Created', new Date(child.created_at).toLocaleDateString()],
              ].map(([k, v]) => (
                <tr key={k}>
                  <td style={{ padding: '5px 8px', color: '#888', whiteSpace: 'nowrap' }}>{k}</td>
                  <td style={{ padding: '5px 8px', color: '#e0e0f0', fontWeight: 500 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ background: '#1e1e2e', borderRadius: 10, padding: '1.25rem' }}>
          <h3 style={{ margin: '0 0 12px', color: '#c4b5fd', fontSize: '1rem' }}>Actions</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <button onClick={() => navigate(`/camera?child=${child.id}`)}
              style={{ ...btnPrimary, width: '100%', background: '#059669' }}>
              🎥 Start New Assessment
            </button>
            <button onClick={() => navigate(`/children/${child.id}/edit`)}
              style={{ ...btnPrimary, width: '100%', background: '#2a2a3e', border: '1px solid #444', color: '#a0a0c0' }}>
              ✏ Edit Profile
            </button>
            <button onClick={handleDeactivate} disabled={!child.is_active}
              style={{ ...btnPrimary, width: '100%', background: '#451a03', border: '1px solid #92400e', color: '#fbbf24', opacity: child.is_active ? 1 : 0.5 }}>
              ⏸ Deactivate Profile
            </button>
            <button onClick={handleDelete} disabled={deleting}
              style={{ ...btnPrimary, width: '100%', background: '#3b0000', border: '1px solid #7f1d1d', color: '#fca5a5' }}>
              🗑 Permanently Delete All Records
            </button>
          </div>
        </div>
      </div>

      {/* Assessment history */}
      <div style={{ background: '#1e1e2e', borderRadius: 10, padding: '1.25rem' }}>
        <h3 style={{ margin: '0 0 12px', color: '#c4b5fd', fontSize: '1rem' }}>📊 Assessment History</h3>
        {assessments.length === 0 ? (
          <p style={{ color: '#888', fontSize: '0.9rem' }}>No assessments yet for this child.</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #333' }}>
                {['Date', 'Age Group', 'Status', 'Model', 'Result', ''].map(h => (
                  <th key={h} style={{ padding: '6px 8px', color: '#888', textAlign: 'left', fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {assessments.map(a => (
                <tr key={a.id} style={{ borderBottom: '1px solid #2a2a3e' }}>
                  <td style={{ padding: '7px 8px', color: '#a0a0c0' }}>{new Date(a.created_at).toLocaleDateString()}</td>
                  <td style={{ padding: '7px 8px' }}>{a.age_group_label}</td>
                  <td style={{ padding: '7px 8px', color: '#c4b5fd' }}>{a.status}</td>
                  <td style={{ padding: '7px 8px', color: '#888', fontSize: '0.8rem' }}>{a.model_version ?? '—'}</td>
                  <td style={{ padding: '7px 8px', fontSize: '0.8rem' }}>
                    {a.result_summary ? JSON.stringify(a.result_summary).slice(0, 40) + '…' : '—'}
                  </td>
                  <td style={{ padding: '7px 8px' }}>
                    {a.job_id && (
                      <button onClick={() => navigate(`/results/${a.job_id}`)}
                        style={{ padding: '3px 8px', background: '#7c3aed', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', fontSize: '0.8rem' }}>
                        View
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
