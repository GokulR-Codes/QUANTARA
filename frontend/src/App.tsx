import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import CameraPage from './pages/CameraPage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import TargetSelectionPage from './pages/TargetSelectionPage';
import ExperimentsPage from './pages/ExperimentsPage';
import SettingsPage from './pages/SettingsPage';
import './App.css';

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-layout">
      <nav className="sidebar" aria-label="Main navigation">
        <div className="sidebar-brand">
          <span className="brand-icon">⚛</span>
          <span className="brand-text">QCFL Research</span>
        </div>
        <ul className="nav-list">
          {[
            { to: '/', label: 'Dashboard', icon: '📊' },
            { to: '/camera', label: 'Camera', icon: '🎥' },
            { to: '/upload', label: 'Upload', icon: '⬆' },
            { to: '/experiments', label: 'Experiments', icon: '🔬' },
            { to: '/settings', label: 'Settings', icon: '⚙' },
          ].map(({ to, label, icon }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={to === '/'}
                className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}
              >
                <span className="nav-icon" aria-hidden="true">{icon}</span>
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
        <div className="sidebar-disclaimer">
          <small>⚠ Experimental research prototype — not a clinical tool</small>
        </div>
      </nav>
      <main className="main-content" id="main-content">
        {children}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/camera" element={<CameraPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/select/:jobId" element={<TargetSelectionPage />} />
          <Route path="/results/:jobId" element={<ResultsPage />} />
          <Route path="/experiments" element={<ExperimentsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
