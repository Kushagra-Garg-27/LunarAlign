import { useState, useEffect } from 'react';
import { checkHealth } from './services/api';
import RegistrationDashboard from './pages/RegistrationDashboard/RegistrationDashboard';
import './App.css';

export default function App() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [healthLoading, setHealthLoading] = useState(true);

  const verifyBackend = async () => {
    setHealthLoading(true);
    try {
      const data = await checkHealth();
      setBackendHealth({ reachable: true, data });
    } catch {
      setBackendHealth({ reachable: false, data: null });
    } finally {
      setHealthLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    checkHealth()
      .then((data) => {
        if (!ignore) setBackendHealth({ reachable: true, data });
      })
      .catch(() => {
        if (!ignore) setBackendHealth({ reachable: false, data: null });
      })
      .finally(() => {
        if (!ignore) setHealthLoading(false);
      });

    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div className="app">
      {/* Top Navigation Bar */}
      <header className="app-navbar">
        <div className="navbar-brand">
          <span className="brand-logo-dot" />
          <div>
            <h1 className="brand-title">SIH26166 · LUNAR REGISTRATION</h1>
            <p className="brand-sub">Chandrayaan-2 High-Resolution Spatial Co-Registration</p>
          </div>
        </div>

        <div className="navbar-status">
          {healthLoading ? (
            <span className="health-badge health-checking">Checking backend…</span>
          ) : backendHealth?.reachable ? (
            <div className="health-badge health-online" title={`Backend ${backendHealth.data?.version || ''} online`}>
              <span className="health-dot dot-green" />
              <span>API ONLINE (8000)</span>
            </div>
          ) : (
            <button
              type="button"
              className="health-badge health-offline"
              onClick={verifyBackend}
              title="Backend unreachable at :8000. Click to retry."
            >
              <span className="health-dot dot-red" />
              <span>API OFFLINE · RETRY</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="main-viewport">
        <div className="dashboard-container">
          <RegistrationDashboard />
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>SIH26166 — ISRO Chandrayaan-2 · TMC-2 / OHRC / IIRS Classical SIFT Baseline (Stage A)</p>
      </footer>
    </div>
  );
}
