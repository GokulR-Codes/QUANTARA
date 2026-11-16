/* Q-CARE Auth pages — Login, Register, OTP, Forgot/Reset password */
import { useState, type FormEvent } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import {
  login as apiLogin, register as apiRegister,
  verifyOtp, resendOtp, forgotPassword, resetPassword,
} from '../api/authClient';
import { apiErrorMessage } from '../api/client';

// ── Shared style helpers ─────────────────────────────────────────────────

const cardStyle: React.CSSProperties = {
  maxWidth: 480, margin: '60px auto', padding: '2.5rem 2rem',
  background: 'var(--surface, #1e1e2e)', borderRadius: 12,
  boxShadow: '0 4px 32px rgba(0,0,0,0.3)',
};
const inputStyle: React.CSSProperties = {
  width: '100%', padding: '10px 12px', borderRadius: 6,
  border: '1px solid #444', background: '#2a2a3e', color: '#fff',
  fontSize: '0.95rem', boxSizing: 'border-box', marginTop: 4,
};
const btnPrimary: React.CSSProperties = {
  width: '100%', padding: '11px', borderRadius: 6, border: 'none',
  background: '#7c3aed', color: '#fff', fontSize: '1rem',
  cursor: 'pointer', marginTop: 8, fontWeight: 600,
};
const btnSecondary: React.CSSProperties = {
  ...btnPrimary, background: 'transparent', border: '1px solid #7c3aed', color: '#a78bfa',
};
const errStyle: React.CSSProperties = {
  background: '#3b0000', color: '#fca5a5', padding: '10px 12px',
  borderRadius: 6, marginTop: 8, fontSize: '0.9rem',
};
const successStyle: React.CSSProperties = {
  ...errStyle, background: '#004d26', color: '#86efac',
};

function FormGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ fontSize: '0.88rem', color: '#a0a0c0' }}>{label}</label>
      {children}
    </div>
  );
}

// ── Tab navigation for auth ──────────────────────────────────────────────

type AuthTab = 'login' | 'register' | 'forgot';

function AuthTabs({ tab, setTab }: { tab: AuthTab; setTab: (t: AuthTab) => void }) {
  return (
    <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
      {(['login', 'register', 'forgot'] as AuthTab[]).map(t => (
        <button
          key={t}
          onClick={() => setTab(t)}
          style={{
            flex: 1, padding: '8px 0', borderRadius: 6, border: 'none',
            background: tab === t ? '#7c3aed' : '#2a2a3e',
            color: tab === t ? '#fff' : '#a0a0c0',
            cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
            fontSize: '0.9rem', textTransform: 'capitalize',
          }}
        >
          {t === 'forgot' ? 'Reset Password' : t.charAt(0).toUpperCase() + t.slice(1)}
        </button>
      ))}
    </div>
  );
}

// ── LOGIN FORM ───────────────────────────────────────────────────────────

function LoginForm() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({ email_or_username: '', password: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const data = await apiLogin(form);
      login(data.access_token, data.guardian);
      navigate(data.guardian.is_verified ? '/guardian' : '/verify-email');
    } catch (err) {
      setError(apiErrorMessage(err, 'Login failed. Check your credentials.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <FormGroup label="Email or Username">
        <input
          style={inputStyle}
          type="text"
          value={form.email_or_username}
          onChange={e => setForm(f => ({ ...f, email_or_username: e.target.value }))}
          required autoFocus autoComplete="username"
          placeholder="email@example.com or username"
        />
      </FormGroup>
      <FormGroup label="Password">
        <input
          style={inputStyle} type="password" value={form.password}
          onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
          required autoComplete="current-password" placeholder="Your password"
        />
      </FormGroup>
      {error && <div style={errStyle}>{error}</div>}
      <button style={btnPrimary} type="submit" disabled={loading}>
        {loading ? 'Signing in…' : 'Sign In'}
      </button>
    </form>
  );
}

// ── REGISTER FORM ────────────────────────────────────────────────────────

function RegisterForm({ onSuccess }: { onSuccess: () => void }) {
  const [form, setForm] = useState({
    email: '', username: '', password: '', confirm_password: '',
    full_name: '', mobile: '', relationship_to_child: '', country: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const f = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm(p => ({ ...p, [k]: e.target.value }));

  const validatePwd = (pwd: string): string | null => {
    if (pwd.length < 8) return 'At least 8 characters required.';
    if (!/[A-Z]/.test(pwd)) return 'Needs an uppercase letter.';
    if (!/[a-z]/.test(pwd)) return 'Needs a lowercase letter.';
    if (!/\d/.test(pwd)) return 'Needs a digit.';
    if (!/[!@#$%^&*()_+\-=[\]{}|;':",.<>?/]/.test(pwd)) return 'Needs a special character.';
    return null;
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    const pwdErr = validatePwd(form.password);
    if (pwdErr) { setError(pwdErr); return; }
    if (form.password !== form.confirm_password) { setError('Passwords do not match.'); return; }
    setLoading(true);
    try {
      await apiRegister({
        email: form.email, username: form.username,
        password: form.password, confirm_password: form.confirm_password,
        full_name: form.full_name,
        mobile: form.mobile || undefined,
        relationship_to_child: form.relationship_to_child || undefined,
        country: form.country || undefined,
      });
      onSuccess();
    } catch (err) {
      setError(apiErrorMessage(err, 'Registration failed.'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={submit}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 12px' }}>
        <FormGroup label="Full Name *">
          <input style={inputStyle} type="text" value={form.full_name} onChange={f('full_name')} required placeholder="Your full name" />
        </FormGroup>
        <FormGroup label="Relationship to Child *">
          <input style={inputStyle} type="text" value={form.relationship_to_child} onChange={f('relationship_to_child')} placeholder="e.g. Parent, Guardian" />
        </FormGroup>
        <FormGroup label="Email Address *">
          <input style={inputStyle} type="email" value={form.email} onChange={f('email')} required placeholder="email@example.com" autoComplete="email" />
        </FormGroup>
        <FormGroup label="Username *">
          <input style={inputStyle} type="text" value={form.username} onChange={f('username')} required placeholder="username" autoComplete="username" minLength={3} />
        </FormGroup>
        <FormGroup label="Password *">
          <input style={inputStyle} type="password" value={form.password} onChange={f('password')} required placeholder="Min 8 chars, mixed" autoComplete="new-password" />
        </FormGroup>
        <FormGroup label="Confirm Password *">
          <input style={inputStyle} type="password" value={form.confirm_password} onChange={f('confirm_password')} required placeholder="Repeat password" autoComplete="new-password" />
        </FormGroup>
        <FormGroup label="Mobile (optional)">
          <input style={inputStyle} type="tel" value={form.mobile} onChange={f('mobile')} placeholder="+1 234 567 8901" />
        </FormGroup>
        <FormGroup label="Country (optional)">
          <input style={inputStyle} type="text" value={form.country} onChange={f('country')} placeholder="Country or region" />
        </FormGroup>
      </div>
      <div style={{ background: '#1a1a2e', borderRadius: 6, padding: '10px 12px', fontSize: '0.82rem', color: '#a0a0c0', marginTop: 4, marginBottom: 8 }}>
        ⚠ Q-CARE is a research prototype, not a clinical tool. Registration must be completed by a parent or legal guardian, not independently by the child.
      </div>
      {error && <div style={errStyle}>{error}</div>}
      <button style={btnPrimary} type="submit" disabled={loading}>
        {loading ? 'Registering…' : 'Create Guardian Account'}
      </button>
    </form>
  );
}

// ── FORGOT / RESET PASSWORD FORM ─────────────────────────────────────────

function ForgotPasswordForm() {
  const [step, setStep] = useState<'request' | 'reset' | 'done'>('request');
  const [email, setEmail] = useState('');
  const [otp, setOtp] = useState('');
  const [newPwd, setNewPwd] = useState('');
  const [confirmPwd, setConfirmPwd] = useState('');
  const [msg, setMsg] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const requestReset = async (e: FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      const r = await forgotPassword(email);
      setMsg(r.message);
      setStep('reset');
    } catch (err) { setError(apiErrorMessage(err, 'Request failed.')); }
    finally { setLoading(false); }
  };

  const doReset = async (e: FormEvent) => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      const r = await resetPassword(email, otp, newPwd, confirmPwd);
      setMsg(r.message);
      setStep('done');
    } catch (err) { setError(apiErrorMessage(err, 'Reset failed.')); }
    finally { setLoading(false); }
  };

  if (step === 'done') return <div style={successStyle}>{msg}</div>;

  return step === 'request' ? (
    <form onSubmit={requestReset}>
      <p style={{ color: '#a0a0c0', marginBottom: 16, fontSize: '0.9rem' }}>
        Enter your registered email. A reset code will be sent (or printed in the terminal in dev mode).
      </p>
      <FormGroup label="Email Address">
        <input style={inputStyle} type="email" value={email} onChange={e => setEmail(e.target.value)} required placeholder="your@email.com" />
      </FormGroup>
      {error && <div style={errStyle}>{error}</div>}
      <button style={btnPrimary} type="submit" disabled={loading}>{loading ? 'Sending…' : 'Send Reset Code'}</button>
    </form>
  ) : (
    <form onSubmit={doReset}>
      {msg && <div style={successStyle} >{msg}</div>}
      <FormGroup label="6-digit Reset Code">
        <input style={inputStyle} type="text" value={otp} onChange={e => setOtp(e.target.value)} required maxLength={6} placeholder="123456" />
      </FormGroup>
      <FormGroup label="New Password">
        <input style={inputStyle} type="password" value={newPwd} onChange={e => setNewPwd(e.target.value)} required placeholder="New password" />
      </FormGroup>
      <FormGroup label="Confirm New Password">
        <input style={inputStyle} type="password" value={confirmPwd} onChange={e => setConfirmPwd(e.target.value)} required placeholder="Repeat new password" />
      </FormGroup>
      {error && <div style={errStyle}>{error}</div>}
      <button style={btnPrimary} type="submit" disabled={loading}>{loading ? 'Resetting…' : 'Reset Password'}</button>
    </form>
  );
}

// ── MAIN AUTH PAGE ───────────────────────────────────────────────────────

export function AuthPage() {
  const [tab, setTab] = useState<AuthTab>('login');
  const [registerDone, setRegisterDone] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLoginAfterReg = async () => {
    // After registration, the user needs to verify email
    navigate('/verify-email');
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#13131f', padding: '1rem' }}>
      <div style={cardStyle}>
        {/* Brand */}
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <div style={{ fontSize: '2.5rem' }}>⚛</div>
          <h1 style={{ margin: '4px 0 2px', fontSize: '1.6rem', color: '#a78bfa' }}>Q-CARE</h1>
          <p style={{ color: '#666', fontSize: '0.82rem', margin: 0 }}>
            Quantum-Centric AI · Child Behavioural Risk Evaluation
          </p>
          <p style={{ color: '#ff6b6b', fontSize: '0.78rem', margin: '6px 0 0' }}>
            ⚠ Research Prototype — Not a clinical tool
          </p>
        </div>

        <AuthTabs tab={tab} setTab={setTab} />

        {tab === 'login' && <LoginForm />}
        {tab === 'register' && !registerDone && (
          <RegisterForm onSuccess={() => { setRegisterDone(true); handleLoginAfterReg(); }} />
        )}
        {tab === 'register' && registerDone && (
          <div style={successStyle}>
            Account created! Check your email (or terminal in dev mode) for a 6-digit verification code.
            <button style={{ ...btnSecondary, marginTop: 12 }} onClick={() => navigate('/verify-email')}>
              Enter Verification Code →
            </button>
          </div>
        )}
        {tab === 'forgot' && <ForgotPasswordForm />}
      </div>
    </div>
  );
}

// ── OTP VERIFICATION PAGE ────────────────────────────────────────────────

export function VerifyEmailPage() {
  const { guardian, updateGuardian, logout } = useAuth();
  const navigate = useNavigate();
  const [otp, setOtp] = useState('');
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);
  const [resending, setResending] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault(); setError(''); setMsg(''); setLoading(true);
    try {
      await verifyOtp(otp);
      if (guardian) updateGuardian({ ...guardian, is_verified: true });
      navigate('/guardian');
    } catch (err) { setError(apiErrorMessage(err, 'Verification failed.')); }
    finally { setLoading(false); }
  };

  const resend = async () => {
    setResending(true); setError(''); setMsg('');
    try {
      const r = await resendOtp();
      setMsg(r.message);
    } catch (err) { setError(apiErrorMessage(err, 'Could not resend.')); }
    finally { setResending(false); }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#13131f' }}>
      <div style={{ ...cardStyle, maxWidth: 400 }}>
        <h2 style={{ color: '#a78bfa', textAlign: 'center', marginBottom: 8 }}>⚛ Verify Your Email</h2>
        <p style={{ color: '#a0a0c0', textAlign: 'center', marginBottom: 20, fontSize: '0.9rem' }}>
          Enter the 6-digit code sent to <strong>{guardian?.email ?? 'your email'}</strong>.<br />
          In development mode, check the backend terminal.
        </p>
        <form onSubmit={submit}>
          <FormGroup label="Verification Code">
            <input
              style={{ ...inputStyle, textAlign: 'center', letterSpacing: '0.4em', fontSize: '1.4rem' }}
              type="text" value={otp} onChange={e => setOtp(e.target.value.replace(/\D/g, '').slice(0, 6))}
              required maxLength={6} placeholder="000000" autoFocus
            />
          </FormGroup>
          {error && <div style={errStyle}>{error}</div>}
          {msg && <div style={successStyle}>{msg}</div>}
          <button style={btnPrimary} type="submit" disabled={loading || otp.length !== 6}>
            {loading ? 'Verifying…' : 'Verify Code'}
          </button>
        </form>
        <button style={{ ...btnSecondary, marginTop: 8 }} onClick={resend} disabled={resending}>
          {resending ? 'Resending…' : 'Resend Code'}
        </button>
        <button style={{ ...btnSecondary, marginTop: 4, background: 'transparent', borderColor: '#444', color: '#888' }} onClick={logout}>
          Log Out
        </button>
      </div>
    </div>
  );
}
