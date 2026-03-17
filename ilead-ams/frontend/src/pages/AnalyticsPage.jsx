import React from 'react';
import { DomainProgressChart } from '../components/Analytics/DomainProgressChart';
import { CompetencyHeatmap } from '../components/Analytics/CompetencyHeatmap';
import '../styles/AnalyticsPage.css';

export const AnalyticsPage = () => {
  return (
    <div className="analytics-page">
      <div className="page-header">
        <h1>Analytics Dashboard</h1>
        <p>View performance metrics and competency assessments</p>
      </div>

      <div className="analytics-grid">
        <div className="chart-card">
          <DomainProgressChart />
        </div>

        <div className="chart-card">
          <CompetencyHeatmap />
        </div>
      </div>
    </div>
  );
};
