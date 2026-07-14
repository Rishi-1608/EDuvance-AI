import { useState } from 'react';
import { Lock, User, LogIn, AlertCircle, Zap, UserPlus } from 'lucide-react';
import { login, register } from '@/lib/api';
import './LoginPage.css';

export function LoginPage({ onLogin }) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isRegister) {
        const data = await register(username, email, password);
        localStorage.setItem('eduvance_ai_token', data.access_token);
        onLogin(data.user);
      } else {
        const data = await login(username, password);
        localStorage.setItem('eduvance_ai_token', data.access_token);
        onLogin(data.user);
      }
    } catch (err) {
      setError(err.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-split-container">
        <div className="login-video-section animate-in">
          <video
            src="/assets/video/Eduvance_AI_logo_202604042112.mp4"
            autoPlay
            loop
            muted
            playsInline
            className="login-video"
          />
        </div>

        <div className="login-form-section">
          <div className="login-glow" />
          <div className="login-card glass animate-up">
            <div className="login-header">
              <div className="login-logo">
                <img src="/logo.png" alt="Logo" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: 'inherit' }} />
              </div>
              <h1 className="gradient-text login-title">EDuvance AI</h1>
              <p className="login-subtitle">Advanced Lecture Intelligence Platform</p>
            </div>

            <form className="login-form" onSubmit={handleSubmit}>
              <div className="input-group">
                <label className="mono input-label">Username</label>
                <div className="input-field-wrapper">
                  <User size={18} className="input-icon" />
                  <input
                    type="text"
                    className="input-field"
                    placeholder="Username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                  />
                </div>
              </div>

              {isRegister && (
                <div className="input-group">
                  <label className="mono input-label">Email</label>
                  <div className="input-field-wrapper">
                    <User size={18} className="input-icon" style={{ opacity: 0.5 }} />
                    <input
                      type="email"
                      className="input-field"
                      placeholder="Email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                    />
                  </div>
                </div>
              )}

              <div className="input-group">
                <label className="mono input-label">Password</label>
                <div className="input-field-wrapper">
                  <Lock size={18} className="input-icon" />
                  <input
                    type="password"
                    className="input-field"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
              </div>

              {error && (
                <div className="login-error animate-in">
                  <AlertCircle size={14} />
                  <span>{error}</span>
                </div>
              )}

              <button className="btn btn-primary login-btn" disabled={loading} type="submit">
                {loading ? (
                  <><div className="spinner" style={{ width: 16, height: 16 }} /> Processing...</>
                ) : isRegister ? (
                  <><UserPlus size={18} /> Create Account</>
                ) : (
                  <><LogIn size={18} /> Sign In</>
                )}
              </button>
            </form>

            <div className="login-footer">
              <p>{isRegister ? "Already have an account?" : "Don't have an account?"}</p>
              <button
                className="login-toggle-btn mono"
                onClick={() => { setIsRegister(!isRegister); setError(''); }}
                type="button"
              >
                {isRegister ? "Sign In" : "Sign Up"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
