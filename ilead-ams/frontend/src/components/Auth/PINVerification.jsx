import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import '../styles/Auth.css';

export const PINVerification = ({ onSuccess, onCancel }) => {
  const { verifyPin, loading, error } = useAuth();
  const [pin, setPin] = useState('');
  const [localError, setLocalError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);

    if (!pin || pin.length !== 6) {
      setLocalError('PIN must be 6 digits');
      return;
    }

    try {
      await verifyPin(pin);
      if (onSuccess) onSuccess();
    } catch (err) {
      setLocalError(err.response?.data?.message || 'Invalid PIN');
    }
  };

  const handlePinChange = (e) => {
    const value = e.target.value.replace(/\D/g, '').slice(0, 6);
    setPin(value);
  };

  const displayError = localError || error;

  return (
    <div className="auth-card">
      <div className="auth-title">Super Admin Verification</div>
      <div className="auth-subtitle">
        Enter your 6-digit PIN to continue
      </div>

      <form onSubmit={handleSubmit}>
        {displayError && (
          <div className="alert alert-danger">{displayError}</div>
        )}

        <div className="form-group">
          <label>PIN Code</label>
          <input
            type="text"
            inputMode="numeric"
            value={pin}
            onChange={handlePinChange}
            placeholder="000000"
            maxLength="6"
            required
            disabled={loading}
            className="pin-input"
          />
        </div>

        <div className="auth-actions">
          <button
            type="submit"
            className={`btn ${loading ? 'btn-loading' : ''}`}
            disabled={loading}
          >
            {loading ? 'Verifying...' : 'Verify PIN'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onCancel}
            disabled={loading}
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
};
