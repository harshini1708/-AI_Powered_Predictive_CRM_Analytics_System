import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { ApprenticeList } from '../components/Apprentices/ApprenticeList';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { exportApprentices } from '../utils/export';

export const ApprenticesPage = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [apprentices, setApprentices] = useState([]);
  const canManage = ['administrator', 'super_admin'].includes(user?.role);

  const handleSelectApprentice = (id) => {
    navigate(`/apprentices/${id}`);
  };

  const handleExport = () => {
    if (apprentices.length === 0) {
      alert('No apprentices to export');
      return;
    }
    exportApprentices(apprentices);
  };

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/apprentices" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Apprentices</h1>
              <p className="page-subtitle">Manage apprenticeship programs</p>
            </div>
            <div className="page-header-right">
              <button className="btn btn-secondary" onClick={handleExport}>
                ↓ Export CSV
              </button>
              {canManage && (
                <button className="btn" onClick={() => navigate('/apprentices/new')}>
                  + New Apprentice
                </button>
              )}
            </div>
          </div>

          <div className="card">
            <ApprenticeList
              onSelectApprentice={handleSelectApprentice}
              onDataChange={setApprentices}
            />
          </div>
        </main>
      </div>
    </div>
  );
};
