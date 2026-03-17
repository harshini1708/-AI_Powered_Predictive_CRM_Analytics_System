import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { LoginForm } from '../components/Auth/LoginForm';
import { RegisterForm } from '../components/Auth/RegisterForm';
import { PINVerification } from '../components/Auth/PINVerification';
import '../components/styles/Auth.css';

export const AuthPage = ({ onAuthSuccess }) => {
  const { user } = useAuth();
  const [authMode, setAuthMode] = useState('login');
  const [needsPINVerification, setNeedsPINVerification] = useState(false);

  if (user && !needsPINVerification) {
    return null;
  }

  return (
    <div className="auth-container">
      <div className="auth-wrapper">
        <div className="auth-header">
          <h1>I-LEAD AMS</h1>
          <p>Apprenticeship Management System</p>
        </div>

        {needsPINVerification ? (
          <PINVerification
            onSuccess={() => {
              setNeedsPINVerification(false);
              onAuthSuccess?.();
            }}
            onCancel={() => setNeedsPINVerification(false)}
          />
        ) : (
          <>
            <div className="auth-tabs">
              <button
                className={`auth-tab ${authMode === 'login' ? 'active' : ''}`}
                onClick={() => setAuthMode('login')}
              >
                Sign In
              </button>
              <button
                className={`auth-tab ${authMode === 'register' ? 'active' : ''}`}
                onClick={() => setAuthMode('register')}
              >
                Create Account
              </button>
            </div>

            {authMode === 'login' ? (
              <LoginForm
                onSuccess={() => {
                  onAuthSuccess?.();
                  // Check if super admin - would need PIN
                  // setNeedsPINVerification(true);
                }}
                onSwitchMode={setAuthMode}
              />
            ) : (
              <RegisterForm
                onSuccess={() => {
                  setAuthMode('login');
                }}
                onSwitchMode={setAuthMode}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};
