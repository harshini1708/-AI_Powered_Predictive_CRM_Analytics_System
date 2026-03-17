import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import '../styles/Auth.css';

const ROLES = ['apprentice', 'supervisor', 'journeyworker', 'administrator'];

export const RegisterForm = ({ onSuccess, onSwitchMode, programs = [] }) => {
  const { register, loading, error } = useAuth();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    password: '',
    confirmPassword: '',
    role: 'apprentice'
  });
  const [localError, setLocalError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError(null);

    // Validation
    if (!formData.firstName || !formData.lastName || !formData.email || !formData.password) {
      setLocalError('All required fields must be filled');
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setLocalError('Passwords do not match');
      return;
    }

    if (formData.password.length < 8) {
      setLocalError('Password must be at least 8 characters');
      return;
    }

    try {
      await register({
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email.toLowerCase(),
        phone: formData.phone || '',
        password: formData.password,
        role: formData.role
      });
      if (onSuccess) onSuccess();
    } catch (err) {
      setLocalError(err.response?.data?.message || 'Registration failed');
    }
  };

  const displayError = localError || error;

  return (
    <div className="auth-card">
      <div className="auth-title">Create Account</div>
      <div className="auth-subtitle">
        Join the Apprenticeship Management System
      </div>

      <form onSubmit={handleSubmit}>
        {displayError && (
          <div className="alert alert-danger">{displayError}</div>
        )}

        <div className="form-row">
          <div className="form-group">
            <label>First Name *</label>
            <input
              type="text"
              name="firstName"
              value={formData.firstName}
              onChange={handleChange}
              placeholder="John"
              required
              disabled={loading}
            />
          </div>
          <div className="form-group">
            <label>Last Name *</label>
            <input
              type="text"
              name="lastName"
              value={formData.lastName}
              onChange={handleChange}
              placeholder="Doe"
              required
              disabled={loading}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Email Address *</label>
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="john@example.com"
            required
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label>Phone Number</label>
          <input
            type="tel"
            name="phone"
            value={formData.phone}
            onChange={handleChange}
            placeholder="(610) 555-1234"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label>Role *</label>
          <select
            name="role"
            value={formData.role}
            onChange={handleChange}
            required
            disabled={loading}
          >
            {ROLES.map(role => (
              <option key={role} value={role}>
                {role.charAt(0).toUpperCase() + role.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Password *</label>
          <input
            type="password"
            name="password"
            value={formData.password}
            onChange={handleChange}
            placeholder="••••••••"
            required
            disabled={loading}
          />
          <small>Minimum 8 characters</small>
        </div>

        <div className="form-group">
          <label>Confirm Password *</label>
          <input
            type="password"
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleChange}
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
          {loading ? 'Creating Account...' : 'Create Account'}
        </button>
      </form>

      <div className="auth-footer">
        <p>
          Already have an account?{' '}
          <button
            type="button"
            className="link-button"
            onClick={() => onSwitchMode?.('login')}
          >
            Sign in
          </button>
        </p>
      </div>
    </div>
  );
};
