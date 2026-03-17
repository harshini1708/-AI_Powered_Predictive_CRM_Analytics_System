import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './hooks/useAuth';
import { AuthPage } from './pages/AuthPage';
import { DashboardPage } from './pages/DashboardPage';
import { ApprenticesPage } from './pages/ApprenticesPage';
import { ApprenticeDetailPage } from './pages/ApprenticeDetailPage';
import { HourLogsPage } from './pages/HourLogsPage';
import { SubmitHoursPage } from './pages/SubmitHoursPage';
import { ApprovalsPage } from './pages/ApprovalsPage';
import { UsersPage } from './pages/UsersPage';
import { AnalyticsPage } from './pages/AnalyticsPage';

export const Router = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div className="app-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <Routes>
        {!isAuthenticated ? (
          <>
            <Route path="/" element={<AuthPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </>
        ) : (
          <>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/apprentices" element={<ApprenticesPage />} />
            <Route path="/apprentices/:id" element={<ApprenticeDetailPage />} />
            <Route path="/hours" element={<HourLogsPage />} />
            <Route path="/hours/submit" element={<SubmitHoursPage />} />
            <Route path="/approvals" element={<ApprovalsPage />} />
            <Route path="/users" element={<UsersPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </>
        )}
      </Routes>
    </BrowserRouter>
  );
};
