import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import * as apprenticeApi from '../api/apprentices';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { ApprenticeProgress } from '../components/Apprentices/ApprenticeProgress';

export const ApprenticeDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { data, loading, error, execute } = useApi(apprenticeApi.getApprentice);
  const [apprentice, setApprentice] = useState(null);

  useEffect(() => {
    execute(id);
  }, [id, execute]);

  useEffect(() => {
    if (data?.apprentice) {
      setApprentice(data.apprentice);
    }
  }, [data]);

  if (loading) {
    return (
      <div className="app-main">
        <Navbar />
        <div className="main-layout">
          <Sidebar activeRoute="/apprentices" />
          <main className="main-content">
            <div className="alert alert-info">Loading...</div>
          </main>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="app-main">
        <Navbar />
        <div className="main-layout">
          <Sidebar activeRoute="/apprentices" />
          <main className="main-content">
            <div className="alert alert-danger">Error: {error}</div>
          </main>
        </div>
      </div>
    );
  }

  if (!apprentice) {
    return (
      <div className="app-main">
        <Navbar />
        <div className="main-layout">
          <Sidebar activeRoute="/apprentices" />
          <main className="main-content">
            <div className="alert alert-info">Apprentice not found</div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/apprentices" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">{apprentice.name}</h1>
              <p className="page-subtitle">{apprentice.program}</p>
            </div>
            <div className="page-actions">
              <button className="btn btn-secondary" onClick={() => navigate(-1)}>
                ← Back
              </button>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Apprentice Information</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <p><strong>Email:</strong> {apprentice.email}</p>
                <p><strong>Program:</strong> {apprentice.program}</p>
                <p><strong>Status:</strong> <span className={`badge badge-${apprentice.status.toLowerCase()}`}>{apprentice.status}</span></p>
              </div>
              <div>
                <p><strong>Supervisor:</strong> {apprentice.supervisor || '—'}</p>
                <p><strong>Employer:</strong> {apprentice.employer || '—'}</p>
                <p><strong>Start Date:</strong> {apprentice.startDate ? new Date(apprentice.startDate).toLocaleDateString() : '—'}</p>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Progress & Competency</div>
            <ApprenticeProgress apprenticeId={id} />
          </div>
        </main>
      </div>
    </div>
  );
};
