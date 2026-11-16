// Guardian Dashboard — the main landing page after login
import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { listChildren, listAssessments, getAgeGroupStatuses } from '../api/authClient';
import { getReadiness } from '../api/client';
import type { ChildProfile, AssessmentSession, AgeGroupModelStatus } from '../types/auth';
import type { ReadinessResponse } from '../types';

const cardStyle: React.CSSProperties = {
  background: '#1e1e2e', borderRadius: 10, padding: '1.25rem 1.5rem',
  boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
};

const badgeStyle = (color: string): React.CSSProperties => ({
  display: 'inline-block', padding: '2px 10px', borderRadius: 20,
  background: color, fontSize: '0.78rem', fontWeight: 600,
});

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return <span style={badgeStyle(ok ? '#1a4731' : '#3b1f00')}>{ok ? '✅' : '⚠'} {label}</span>;
}

export default function GuardianDashboard() {
  const { guardian, logout } = useAuth();
  const navigate = useNavigate();
  const [children, setChildren] = useState<ChildProfile[]>([]);
  const [recentAssessments, setRecentAssessments] = useState<AssessmentSession[]>([]);
  const [ageModels, setAgeModels] = useState<AgeGroupModelStatus[]>([]);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      listChildren().catch(() => [] as ChildProfile[]),
      listAssessments().catch(() => [] as AssessmentSession[]),
      getAgeGroupStatuses().catch(() => [] as AgeGroupModelStatus[]),
      getReadiness().catch(() => null),
    ]).then(([ch, as_, ag, rd]) => {
      setChildren(ch);
      setRecentAssessments(as_.slice(0, 5));
      setAgeModels(ag);
      setReadiness(rd);
      setLoading(false);
    });
  }, []);

  const handleLogout = async () => { await logout(); navigate('/auth'); };

  return (
    <div style={{ padding: '2rem', maxWidth: 1100, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ margin: 0, color: '#a78bfa', fontSize: '1.8rem' }}>⚛ Q-CARE Guardian Dashboard</h1>
          <p style={{ color: '#888', margin: '4px 0 0', fontSize: '0.9rem' }}>
            Welcome, <strong style={{ color: '#c4b5fd' }}>{guardian?.full_name}</strong>
            {!guardian?.is_verified && (
              <Link to="/verify-email" style={{ marginLeft: 12, color: '#fbbf24', fontSize: '0.85rem' }}>
                ⚠ Email not verified — click to verify
              </Link>
            )}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => navigate('/children/new')}
            style={{ padding: '8px 16px', background: '#7c3aed', border: 'none', borderRadius: 6, color: '#fff', cursor: 'pointer', fontWeight: 600 }}>
            + Add Child Profile
          </button>
          <button onClick={handleLogout}
            style={{ padding: '8px 16px', background: '#2a2a3e', border: '1px solid #444', borderRadius: 6, color: '#a0a0c0', cursor: 'pointer' }}>
            Log Out
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', color: '#888', padding: '3rem' }}>Loading dashboard…</div>
      ) : (
        <>
          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Child Profiles', value: children.length, icon: '👶', color: '#7c3aed' },
              { label: 'Total Assessments', value: recentAssessments.length, icon: '📊', color: '#0891b2' },
              { label: 'Models Ready', value: ageModels.filter(m => m.has_active_model).length + '/3', icon: '🤖', color: '#059669' },
              { label: 'System', value: readiness?.ready ? 'Ready' : 'Partial', icon: '⚡', color: readiness?.ready ? '#059669' : '#d97706' },
            ].map(({ label, value, icon, color }) => (
              <div key={label} style={{ ...cardStyle, textAlign: 'center' }}>
                <div style={{ fontSize: '2rem' }}>{icon}</div>
                <div style={{ fontSize: '1.6rem', fontWeight: 700, color }}>{value}</div>
                <div style={{ color: '#888', fontSize: '0.85rem' }}>{label}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            {/* Child profiles */}
            <div style={cardStyle}>
              <h3 style={{ margin: '0 0 16px', color: '#c4b5fd' }}>👶 Child Profiles</h3>
              {children.length === 0 ? (
                <div style={{ textAlign: 'center', padding: '1.5rem', color: '#888' }}>
                  <p>No child profiles yet.</p>
                  <button onClick={() => navigate('/children/new')}
                    style={{ padding: '8px 16px', background: '#7c3aed', border: 'none', borderRadius: 6, color: '#fff', cursor: 'pointer' }}>
                    Add First Child Profile
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {children.map(child => (
                    <div key={child.id} style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '10px 12px', background: '#13131f', borderRadius: 8, cursor: 'pointer',
                    }}
                      onClick={() => navigate(`/children/${child.id}`)}>
                      <div>
                        <div style={{ fontWeight: 600 }}>{child.display_name}</div>
                        <div style={{ fontSize: '0.82rem', color: '#888' }}>{child.age_group_label} · {child.assessment_count} assessment(s)</div>
                      </div>
                      <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                        <button onClick={e => { e.stopPropagation(); navigate(`/camera?child=${child.id}`); }}
                          style={{ padding: '4px 10px', background: '#059669', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', fontSize: '0.8rem' }}>
                          🎥 Record
                        </button>
                        <button onClick={e => { e.stopPropagation(); navigate(`/upload?child=${child.id}`); }}
                          style={{ padding: '4px 10px', background: '#4f46e5', border: 'none', borderRadius: 4, color: '#fff', cursor: 'pointer', fontSize: '0.8rem' }}>
                          ⬆ Upload
                        </button>
                        <button onClick={e => { e.stopPropagation(); navigate(`/children/${child.id}`); }}
                          style={{ padding: '4px 10px', background: '#2a2a3e', border: '1px solid #444', borderRadius: 4, color: '#a0a0c0', cursor: 'pointer', fontSize: '0.8rem' }}>
                          View
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Age model status */}
            <div style={cardStyle}>
              <h3 style={{ margin: '0 0 16px', color: '#c4b5fd' }}>🤖 Age-Specific Model Status</h3>
              {ageModels.map(m => (
                <div key={m.age_group} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '10px 12px', background: '#13131f', borderRadius: 8, marginBottom: 8,
                }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{m.label}</div>
                    <div style={{ fontSize: '0.78rem', color: '#888' }}>
                      {m.has_active_model ? `v${m.model_version} · approved` : 'No trained model available'}
                    </div>
                  </div>
                  <StatusBadge ok={m.has_active_model} label={m.has_active_model ? 'Ready' : 'Not ready'} />
                </div>
              ))}
              <div style={{ fontSize: '0.78rem', color: '#666', marginTop: 8 }}>
                Models require labelled training data. Processing without a model will complete up to feature extraction.
              </div>
            </div>

            {/* Recent assessments */}
            <div style={cardStyle}>
              <h3 style={{ margin: '0 0 16px', color: '#c4b5fd' }}>📊 Recent Assessments</h3>
              {recentAssessments.length === 0 ? (
                <p style={{ color: '#888', fontSize: '0.9rem' }}>No assessments yet.</p>
              ) : (
                recentAssessments.map(a => (
                  <div key={a.id} style={{
                    padding: '8px 12px', background: '#13131f', borderRadius: 8, marginBottom: 6,
                    display: 'flex', justifyContent: 'space-between',
                  }}>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{a.child_name ?? 'Unknown'}</div>
                      <div style={{ fontSize: '0.78rem', color: '#888' }}>{a.age_group_label} · {a.status}</div>
                    </div>
                    <div style={{ fontSize: '0.78rem', color: '#666' }}>
                      {new Date(a.created_at).toLocaleDateString()}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* System status */}
            <div style={cardStyle}>
              <h3 style={{ margin: '0 0 16px', color: '#c4b5fd' }}>⚡ System Status</h3>
              {readiness?.components.map(c => (
                <div key={c.name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: '0.88rem', color: '#a0a0c0' }}>{c.name}</span>
                  <StatusBadge ok={c.status === 'ok'} label={c.status} />
                </div>
              ))}
              {readiness?.message && (
                <p style={{ color: '#888', fontSize: '0.8rem', marginTop: 8 }}>{readiness.message}</p>
              )}
            </div>
          </div>

          {/* Disclaimer */}
          <div style={{ marginTop: 20, padding: '12px 16px', background: '#1a1a0e', borderRadius: 8, borderLeft: '3px solid #d97706' }}>
            <p style={{ margin: 0, color: '#fbbf24', fontSize: '0.82rem' }}>
              ⚠ <strong>Research Prototype Disclaimer:</strong> Q-CARE is an experimental research system.
              Assessment results are not medical or developmental diagnoses. All data is stored locally on this device.
              Do not use for clinical decision-making.
            </p>
          </div>
        </>
      )}
    </div>
  );
}
