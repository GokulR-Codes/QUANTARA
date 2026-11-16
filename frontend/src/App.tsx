import { BrowserRouter, Routes, Route, NavLink, useNavigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';

// Existing pages (preserved)
import CameraPage from './pages/CameraPage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import TargetSelectionPage from './pages/TargetSelectionPage';
import ExperimentsPage from './pages/ExperimentsPage';
import SettingsPage from './pages/SettingsPage';
import Dashboard from './pages/Dashboard';

// Q-CARE upgrade pages
import { AuthPage, VerifyEmailPage } from './pages/AuthPage';
import GuardianDashboard from './pages/GuardianDashboard';
import { CreateChildPage, ChildProfilePage } from './pages/ChildProfilePage';
import PrivacyPage from './pages/PrivacyPage';
import AgeModelsPage from './pages/AgeModelsPage';

import './App.css';

// ── Authenticated sidebar layout ─────────────────────────────────────────

function AuthenticatedLayout({ children }: { children: React.ReactNode }) {
  const { guardian, logout } = useAuth();
  const navigate = useNavigate();

  const navItems = [
    { to: '/guardian', label: 'Dashboard', icon: '🏠', end: true },
    { to: '/camera', label: 'Assessment', icon: '🎥' },
    { to: '/upload', label: 'Upload Video', icon: '⬆' },
    { to: '/age-models', label: 'Age Models', icon: '🧬' },
    { to: '/experiments', label: 'Experiments', icon: '🔬' },
    { to: '/settings', label: 'Settings', icon: '⚙' },
    { to: '/privacy', label: 'Privacy', icon: '🔒' },
  ];

  return (
    <div className="app-layout">
      <nav className="sidebar" aria-label="Main navigation">
        <div className="sidebar-brand">
          <span className="brand-icon">⚛</span>
          <span className="brand-text">Q-CARE</span>
        </div>

        {/* Guardian info */}
        {guardian && (
          <div style={{
            padding: '10px 16px', margin: '0 0 8px',
            background: '#13131f', borderRadius: 8, fontSize: '0.82rem',
          }}>
            <div style={{ color: '#c4b5fd', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {guardian.full_name}
            </div>
            <div style={{ color: '#666', fontSize: '0.75rem' }}>
              {guardian.is_verified ? '✅ Verified' : '⚠ Unverified'}
            </div>
          </div>
        )}

        <ul className="nav-list">
          {navItems.map(({ to, label, icon, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
              >
                <span className="nav-icon" aria-hidden="true">{icon}</span>
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div style={{ marginTop: 'auto', padding: '12px 16px' }}>
          <button
            onClick={async () => { await logout(); navigate('/auth'); }}
            style={{
              width: '100%', padding: '8px', borderRadius: 6, border: '1px solid #444',
              background: 'transparent', color: '#888', cursor: 'pointer', fontSize: '0.85rem',
            }}
          >
            Log Out
          </button>
        </div>

        <div className="sidebar-disclaimer">
          <small>⚠ Research prototype — not a clinical tool</small>
        </div>
      </nav>
      <main className="main-content" id="main-content">
        {children}
      </main>
    </div>
  );
}

// ── Root router ───────────────────────────────────────────────────────────

function AppRoutes() {
  return (
    <Routes>
      {/* Public auth routes */}
      <Route path="/auth" element={<AuthPage />} />
      <Route path="/verify-email" element={<VerifyEmailPage />} />

      {/* Root redirect — if logged in go to guardian dashboard, else auth */}
      <Route path="/" element={
        <ProtectedRoute requireVerified={false}>
          <AuthenticatedLayout>
            <GuardianDashboard />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Guardian dashboard (requires auth + verified) */}
      <Route path="/guardian" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <GuardianDashboard />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Child profile management */}
      <Route path="/children/new" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <CreateChildPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
      <Route path="/children/:childId" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <ChildProfilePage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Assessment/video workflow (preserved, now auth-protected) */}
      <Route path="/camera" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <CameraPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
      <Route path="/upload" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <UploadPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
      <Route path="/select/:jobId" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <TargetSelectionPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
      <Route path="/results/:jobId" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <ResultsPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Research tools */}
      <Route path="/experiments" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <ExperimentsPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Settings & privacy */}
      <Route path="/settings" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <SettingsPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
      <Route path="/privacy" element={
        <AuthenticatedLayout>
          <PrivacyPage />
        </AuthenticatedLayout>
      } />

      {/* Age-specific models */}
      <Route path="/age-models" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <AgeModelsPage />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />

      {/* Legacy research dashboard (unauthenticated) */}
      <Route path="/research" element={
        <ProtectedRoute>
          <AuthenticatedLayout>
            <Dashboard />
          </AuthenticatedLayout>
        </ProtectedRoute>
      } />
    </Routes>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}
