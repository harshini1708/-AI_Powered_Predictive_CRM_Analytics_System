import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as analyticsApi from '../../api/analytics';
import '../styles/Dashboard.css';

export const DashboardStats = () => {
  const { data, loading, error, execute } = useApi(analyticsApi.getDashboardStats);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    execute();
  }, [execute]);

  useEffect(() => {
    if (data?.dashboard) {
      setStats(data.dashboard);
    }
  }, [data]);

  if (loading) {
    return (
      <div className="dashboard-stats">
        <div className="stats-grid">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="stat-card skeleton">
              <div className="stat-skeleton"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="alert alert-danger">
        Failed to load dashboard stats: {error}
      </div>
    );
  }

  if (!stats) {
    return <div className="alert alert-info">No data available</div>;
  }

  return (
    <div className="dashboard-stats">
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Users</div>
          <div className="stat-value">{stats.totalUsers || 0}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Active Apprentices</div>
          <div className="stat-value">{stats.activeApprentices || 0}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Total OJT Hours</div>
          <div className="stat-value">{(stats.totalOjtHours || 0).toFixed(1)}</div>
          <div className="stat-sub">hours logged</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Qualified Hours</div>
          <div className="stat-value">{(stats.qualifiedOjt || 0).toFixed(1)}</div>
          <div className="stat-sub">hours approved</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Pending Approvals</div>
          <div className="stat-value">{stats.pendingApprovals || 0}</div>
          <div className="stat-sub">awaiting review</div>
        </div>
      </div>
    </div>
  );
};
