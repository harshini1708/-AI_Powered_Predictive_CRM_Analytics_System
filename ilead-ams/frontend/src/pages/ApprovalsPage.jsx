import React, { useState } from 'react';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { HourLogList } from '../components/HourLogs/HourLogList';
import { ApprovalForm } from '../components/HourLogs/ApprovalForm';

export const ApprovalsPage = () => {
  const [selectedLogId, setSelectedLogId] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleApprovalSuccess = () => {
    setSelectedLogId(null);
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/approvals" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Hour Approvals</h1>
              <p className="page-subtitle">Review and approve submitted hours</p>
            </div>
          </div>

          {selectedLogId ? (
            <div className="card">
              <div style={{ marginBottom: '20px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setSelectedLogId(null)}
                >
                  ← Back to List
                </button>
              </div>
              <ApprovalForm
                logId={selectedLogId}
                onSuccess={handleApprovalSuccess}
                onCancel={() => setSelectedLogId(null)}
              />
            </div>
          ) : (
            <div className="card">
              <div className="card-title">Pending Hour Logs</div>
              <HourLogList
                onSelectLog={setSelectedLogId}
                key={refreshTrigger}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
