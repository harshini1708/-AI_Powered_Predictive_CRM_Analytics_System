import React, { useEffect, useState } from 'react';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { useApi } from '../hooks/useApi';
import * as adminApi from '../api/admin';

export const AdminDashboardPage = () => {
  const { data: systemHealth, execute: getSystemHealth } = useApi(adminApi.getSystemHealth);
  const { data: userStats, execute: getUserStats } = useApi(adminApi.getUserStatistics);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    getSystemHealth();
    getUserStats();
  }, [getSystemHealth, getUserStats]);

  useEffect(() => {
    if (systemHealth) {
      setStats(systemHealth);
    }
  }, [systemHealth]);

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/admin" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Admin Dashboard</h1>
              <p className="page-subtitle">System health and statistics</p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '30px' }}>
            <div style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <h4 style={{ margin: '0 0 15px 0', color: '#003d82' }}>📊 System Status</h4>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: stats?.status === 'healthy' ? '#28a745' : '#dc3545', marginBottom: '10px' }}>
                {stats?.status === 'healthy' ? '✅ Healthy' : '⚠️ Warning'}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                Last checked: {stats?.lastCheck ? new Date(stats.lastCheck).toLocaleTimeString() : 'N/A'}
              </div>
            </div>

            <div style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <h4 style={{ margin: '0 0 15px 0', color: '#003d82' }}>👥 Total Users</h4>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#003d82', marginBottom: '10px' }}>
                {userStats?.totalUsers || 0}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                Active: {userStats?.activeUsers || 0}
              </div>
            </div>

            <div style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <h4 style={{ margin: '0 0 15px 0', color: '#003d82' }}>🎓 Apprentices</h4>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#003d82', marginBottom: '10px' }}>
                {userStats?.totalApprentices || 0}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                Active: {userStats?.activeApprentices || 0}
              </div>
            </div>

            <div style={{
              background: '#f8f9fa',
              padding: '20px',
              borderRadius: '8px',
              border: '1px solid #ddd'
            }}>
              <h4 style={{ margin: '0 0 15px 0', color: '#003d82' }}>📋 Hour Logs</h4>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#003d82', marginBottom: '10px' }}>
                {userStats?.totalHourLogs || 0}
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                Pending: {userStats?.pendingApprovals || 0}
              </div>
            </div>
          </div>

          <div style={{
            background: 'white',
            padding: '20px',
            borderRadius: '8px',
            border: '1px solid #ddd'
          }}>
            <h3 style={{ color: '#003d82', marginTop: 0 }}>⚙️ System Settings</h3>
            <p style={{ color: '#666' }}>
              Quick access to common administration tasks:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
              <a href="/admin/users" style={{
                padding: '12px',
                background: '#003d82',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '4px',
                textAlign: 'center'
              }}>👥 Manage Users</a>
              <a href="/admin/apprentices" style={{
                padding: '12px',
                background: '#003d82',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '4px',
                textAlign: 'center'
              }}>🎓 Manage Apprentices</a>
              <a href="/admin/settings" style={{
                padding: '12px',
                background: '#003d82',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '4px',
                textAlign: 'center'
              }}>⚙️ System Settings</a>
              <a href="/admin/audit" style={{
                padding: '12px',
                background: '#003d82',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '4px',
                textAlign: 'center'
              }}>📊 Audit Logs</a>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};
