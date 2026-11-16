// Protected route — redirects to /auth if not authenticated
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import type { ReactNode } from 'react';

interface Props {
  children: ReactNode;
  requireVerified?: boolean;
}

export default function ProtectedRoute({ children, requireVerified = true }: Props) {
  const { isAuthenticated, isVerified, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh', background: '#13131f' }}>
        <div style={{ color: '#a78bfa', fontSize: '1.1rem' }}>⚛ Loading Q-CARE…</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/auth" state={{ from: location }} replace />;
  }

  if (requireVerified && !isVerified) {
    return <Navigate to="/verify-email" replace />;
  }

  return <>{children}</>;
}
