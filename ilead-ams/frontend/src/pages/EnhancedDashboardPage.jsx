import React, { useEffect, useState } from 'react';
import { useTheme } from '../context/ThemeContext';
import { Card, CardGrid } from '../components/UI/Card';
import { Button } from '../components/UI/Button';
import { Badge } from '../components/UI/Alert';
import { useApi } from '../hooks/useApi';
import * as adminApi from '../api/admin';

/**
 * Enhanced Dashboard with improved layout and visualizations
 */
export const EnhancedDashboardPage = () => {
  const { currentTheme, isDarkMode, toggleTheme } = useTheme();
  const { data: stats, execute: loadStats } = useApi(adminApi.getUserStatistics);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
    setLoading(false);
  }, [loadStats]);

  const pageStyle = {
    background: currentTheme.background,
    color: currentTheme.text,
    minHeight: '100vh',
    padding: '20px'
  };

  const headerStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '30px',
    paddingBottom: '20px',
    borderBottom: `1px solid ${currentTheme.border}`
  };

  const statCardStyle = (icon, color) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '15px'
  });

  const iconStyle = (color) => ({
    fontSize: '32px',
    background: color,
    width: '60px',
    height: '60px',
    borderRadius: '10px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  });

  const StatCard = ({ icon, label, value, trend, color }) => (
    <Card padding="lg" hover>
      <div style={statCardStyle(icon, color)}>
        <div style={iconStyle(color)}>
          {icon}
        </div>
        <div style={{ flex: 1 }}>
          <p style={{ margin: '0 0 8px 0', color: currentTheme.textSecondary, fontSize: '12px' }}>
            {label}
          </p>
          <h3 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold', color: currentTheme.text }}>
            {value}
          </h3>
          {trend && (
            <p style={{
              margin: '4px 0 0 0',
              fontSize: '12px',
              color: trend > 0 ? currentTheme.success : currentTheme.danger
            }}>
              {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from last month
            </p>
          )}
        </div>
      </div>
    </Card>
  );

  const ProgressBar = ({ label, value, max, color }) => (
    <div style={{ marginBottom: '15px' }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        marginBottom: '6px'
      }}>
        <span style={{ fontSize: '13px', fontWeight: '500' }}>{label}</span>
        <span style={{ fontSize: '13px', color: currentTheme.textSecondary }}>
          {value}/{max}
        </span>
      </div>
      <div style={{
        width: '100%',
        height: '8px',
        background: currentTheme.gray300,
        borderRadius: '4px',
        overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          width: `${(value / max) * 100}%`,
          background: color,
          transition: 'width 0.3s ease'
        }} />
      </div>
    </div>
  );

  return (
    <div style={pageStyle}>
      {/* Header */}
      <div style={headerStyle}>
        <div>
          <h1 style={{ margin: '0 0 4px 0', fontSize: '28px', fontWeight: 'bold' }}>
            📊 Dashboard
          </h1>
          <p style={{ margin: 0, color: currentTheme.textSecondary }}>
            Welcome back! Here's your system overview.
          </p>
        </div>
        <Button
          variant="outline"
          onClick={toggleTheme}
        >
          {isDarkMode ? '☀️ Light' : '🌙 Dark'}
        </Button>
      </div>

      {/* Key Metrics */}
      <div style={{ marginBottom: '30px' }}>
        <h2 style={{ marginBottom: '15px', color: currentTheme.text }}>Key Metrics</h2>
        <CardGrid>
          <StatCard
            icon="👥"
            label="Total Users"
            value={stats?.totalUsers || 0}
            trend={12}
            color={currentTheme.primary}
          />
          <StatCard
            icon="🎓"
            label="Active Apprentices"
            value={stats?.activeApprentices || 0}
            trend={8}
            color={currentTheme.success}
          />
          <StatCard
            icon="📋"
            label="Pending Approvals"
            value={stats?.pendingApprovals || 0}
            trend={-5}
            color={currentTheme.warning}
          />
        </CardGrid>
      </div>

      {/* Recent Activity */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        {/* Activity List */}
        <Card title="Recent Activity" padding="lg">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              { user: 'John Doe', action: 'submitted hours', time: '2 hours ago', status: 'pending' },
              { user: 'Jane Smith', action: 'approved hours', time: '4 hours ago', status: 'approved' },
              { user: 'Bob Johnson', action: 'rejected hours', time: '1 day ago', status: 'rejected' }
            ].map((item, idx) => (
              <div
                key={idx}
                style={{
                  padding: '12px',
                  background: currentTheme.backgroundSecondary,
                  borderRadius: '6px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <div>
                  <p style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: '500' }}>
                    {item.user}
                  </p>
                  <p style={{
                    margin: 0,
                    fontSize: '12px',
                    color: currentTheme.textSecondary
                  }}>
                    {item.action} {item.time}
                  </p>
                </div>
                <Badge variant={item.status}>
                  {item.status}
                </Badge>
              </div>
            ))}
          </div>
        </Card>

        {/* Quick Stats */}
        <Card title="Progress" padding="lg">
          <ProgressBar
            label="Hour Approvals"
            value={75}
            max={100}
            color={currentTheme.success}
          />
          <ProgressBar
            label="System Health"
            value={95}
            max={100}
            color={currentTheme.primary}
          />
          <ProgressBar
            label="Database Usage"
            value={45}
            max={100}
            color={currentTheme.warning}
          />
          <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: `1px solid ${currentTheme.border}` }}>
            <Button variant="outline" size="sm" style={{ width: '100%' }}>
              View Details
            </Button>
          </div>
        </Card>
      </div>

      {/* Quick Actions */}
      <div style={{ marginTop: '30px' }}>
        <h2 style={{ marginBottom: '15px', color: currentTheme.text }}>Quick Actions</h2>
        <Card padding="lg">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '12px' }}>
            <Button variant="primary">📝 Create Report</Button>
            <Button variant="secondary">👥 Manage Users</Button>
            <Button variant="success">✅ Approve Hours</Button>
            <Button variant="danger">⚙️ Settings</Button>
          </div>
        </Card>
      </div>
    </div>
  );
};
