// Global authentication context — provides guardian state to all pages
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { Guardian } from '../types/auth';
import { clearSession, loadSession, saveSession, logout as apiLogout } from '../api/authClient';

interface AuthContextValue {
  guardian: Guardian | null;
  token: string | null;
  isAuthenticated: boolean;
  isVerified: boolean;
  isLoading: boolean;
  login: (token: string, guardian: Guardian) => void;
  logout: () => Promise<void>;
  updateGuardian: (g: Guardian) => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [guardian, setGuardian] = useState<Guardian | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const { token: t, guardian: g } = loadSession();
    if (t && g) {
      setToken(t);
      setGuardian(g);
    }
    setIsLoading(false);
  }, []);

  const login = (newToken: string, newGuardian: Guardian) => {
    saveSession(newToken, newGuardian);
    setToken(newToken);
    setGuardian(newGuardian);
  };

  const logout = async () => {
    try { await apiLogout(); } catch { /* ignore */ }
    clearSession();
    setToken(null);
    setGuardian(null);
  };

  const updateGuardian = (g: Guardian) => {
    setGuardian(g);
    localStorage.setItem('qcare_guardian', JSON.stringify(g));
  };

  return (
    <AuthContext.Provider value={{
      guardian,
      token,
      isAuthenticated: !!token && !!guardian,
      isVerified: !!guardian?.is_verified,
      isLoading,
      login,
      logout,
      updateGuardian,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
