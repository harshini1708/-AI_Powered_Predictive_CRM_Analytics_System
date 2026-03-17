import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import '../styles/Auth.css';

export const LoginForm = ({ onSuccess, onSwitchMode }) => {
  const { login, loading, error } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [localError, setLocalError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);

    if (!email || !password) {
      setLocalError('Email and password are required');
      return;
    }

    try {
      await login(email, password);
      if (onSuccess) onSuccess();
    } catch (err) {
      setLocalError(err.response?.data?.message || 'Login failed');
    }
  };

  const displayError = localError || error;

  return (
    <div className="auth-card">
      <div className="auth-title">Sign In</div>
      <div className="auth-subtitle">
        Apprenticeship Management System
      </div>

      <form onSubmit={handleSubmit}>
        {displayError && (
          <div className="alert alert-danger">{displayError}</div>
        )}

        <div className="form-group">
          <label>Email Address</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="your@example.com"
            required
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
            disabled={loading}
          />
        </div>

        <button
          type="submit"
          className={`btn ${loading ? 'btn-loading' : ''}`}
          disabled={loading}
        >
          {loading ? 'Signing in...' : 'Sign In'}
        </button>
      </form>

      <div className="auth-footer">
        <p>
          Don't have an account?{' '}
          <button
            type="button"
            className="link-button"
            onClick={() => onSwitchMode?.('register')}
          >
            Create one
          </button>
        </p>
      </div>
    </div>
  );
};
