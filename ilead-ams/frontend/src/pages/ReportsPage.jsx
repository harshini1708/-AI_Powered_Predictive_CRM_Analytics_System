import React, { useEffect, useState } from 'react';
import { useApi } from '../hooks/useApi';
import * as reportApi from '../api/reports';
import { ReportBuilder } from '../components/Reports/ReportBuilder';

export const ReportsPage = () => {
  const { data: reports, execute: loadReports } = useApi(reportApi.getReports);
  const [showBuilder, setShowBuilder] = useState(false);
  const [selectedReport, setSelectedReport] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadReports({ status: 'active' });
  }, [loadReports]);

  const handleCreateReport = async (reportData) => {
    setLoading(true);
    try {
      await reportApi.createReport(reportData);
      setShowBuilder(false);
      await loadReports({ status: 'active' });
    } catch (error) {
      console.error('Error creating report:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteReport = async (reportId) => {
    if (window.confirm('Are you sure you want to delete this report?')) {
      try {
        await reportApi.deleteReport(reportId);
        await loadReports({ status: 'active' });
      } catch (error) {
        alert('Error deleting report: ' + error.message);
      }
    }
  };

  const handleViewReport = (reportId) => {
    window.location.href = `/reports/${reportId}`;
  };

  return (
    <div className="app-main">
      <main className="main-content" style={{ padding: '20px' }}>
        <div className="page-header" style={{ marginBottom: '30px' }}>
          <div className="page-header-left">
            <h1 className="page-title">📊 Reports</h1>
            <p className="page-subtitle">Create and manage custom reports</p>
          </div>
          <button
            onClick={() => setShowBuilder(!showBuilder)}
            style={{
              padding: '10px 20px',
              background: '#003d82',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            {showBuilder ? 'Cancel' : '+ Create Report'}
          </button>
        </div>

        {showBuilder && (
          <div style={{ marginBottom: '30px' }}>
            <ReportBuilder onSave={handleCreateReport} />
          </div>
        )}

        {/* Reports List */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '20px'
        }}>
          {reports && reports.reports && reports.reports.length > 0 ? (
            reports.reports.map(report => (
              <div
                key={report._id}
                style={{
                  background: 'white',
                  padding: '20px',
                  borderRadius: '8px',
                  border: '1px solid #ddd',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
              >
                <h3 style={{ margin: '0 0 10px 0', color: '#003d82' }}>
                  {report.name}
                </h3>
                <p style={{
                  color: '#666',
                  fontSize: '14px',
                  marginBottom: '15px',
                  minHeight: '40px'
                }}>
                  {report.description || 'No description'}
                </p>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '15px',
                  fontSize: '12px',
                  color: '#999'
                }}>
                  <span>📋 {report.type}</span>
                  <span>{new Date(report.createdAt).toLocaleDateString()}</span>
                </div>

                {report.schedule && report.schedule.enabled && (
                  <div style={{
                    background: '#e7f3ff',
                    padding: '8px 12px',
                    borderRadius: '4px',
                    marginBottom: '15px',
                    fontSize: '12px',
                    color: '#0066cc'
                  }}>
                    ⏰ Scheduled: {report.schedule.frequency}
                  </div>
                )}

                <div style={{
                  display: 'flex',
                  gap: '10px'
                }}>
                  <button
                    onClick={() => handleViewReport(report._id)}
                    style={{
                      flex: 1,
                      padding: '10px',
                      background: '#003d82',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '14px'
                    }}
                  >
                    View
                  </button>
                  <button
                    onClick={() => handleDeleteReport(report._id)}
                    style={{
                      flex: 1,
                      padding: '10px',
                      background: '#dc3545',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '14px'
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div style={{
              gridColumn: '1 / -1',
              textAlign: 'center',
              padding: '40px',
              color: '#666'
            }}>
              <p style={{ fontSize: '16px', marginBottom: '10px' }}>No reports yet</p>
              <p style={{ fontSize: '14px' }}>
                Create your first report to get started with custom analytics
              </p>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div style={{
            textAlign: 'center',
            padding: '40px',
            color: '#666'
          }}>
            ⏳ Loading reports...
          </div>
        )}
      </main>
    </div>
  );
};
