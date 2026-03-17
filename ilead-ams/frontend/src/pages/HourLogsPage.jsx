import React, { useState } from 'react';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { HourLogList } from '../components/HourLogs/HourLogList';
import { AdvancedFilters } from '../components/HourLogs/AdvancedFilters';
import { exportHourLogs } from '../utils/export';

export const HourLogsPage = () => {
  const [advancedFilters, setAdvancedFilters] = useState(null);
  const [logs, setLogs] = useState([]);

  const handleApplyFilters = (filters) => {
    setAdvancedFilters(filters);
  };

  const handleResetFilters = () => {
    setAdvancedFilters(null);
  };

  const handleExport = () => {
    if (logs.length === 0) {
      alert('No logs to export');
      return;
    }
    exportHourLogs(logs);
  };

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/hours" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Hour Logs</h1>
              <p className="page-subtitle">View your submitted hours</p>
            </div>
            <div className="page-header-right">
              <button className="btn btn-secondary" onClick={handleExport}>
                ↓ Export CSV
              </button>
            </div>
          </div>

          <AdvancedFilters
            onApplyFilters={handleApplyFilters}
            onReset={handleResetFilters}
          />

          <div className="card">
            <div className="card-title">Your Hour Logs</div>
            <HourLogList
              advancedFilters={advancedFilters}
              onLogsChange={setLogs}
            />
          </div>
        </main>
      </div>
    </div>
  );
};
