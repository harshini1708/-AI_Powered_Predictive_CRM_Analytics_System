import React from 'react';
import { useAuth } from '../hooks/useAuth';
import { DashboardStats } from '../components/Dashboard/DashboardStats';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';

export const DashboardPage = () => {
  const { user } = useAuth();

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/dashboard" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Dashboard</h1>
              <p className="page-subtitle">Welcome back, {user?.firstName}!</p>
            </div>
          </div>
          <DashboardStats />
        </main>
      </div>
    </div>
  );
};
