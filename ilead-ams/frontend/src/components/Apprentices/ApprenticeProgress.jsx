import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as apprenticeApi from '../../api/apprentices';
import '../styles/Progress.css';

export const ApprenticeProgress = ({ apprenticeId }) => {
  const { data, loading, error, execute } = useApi(apprenticeApi.getApprenticeProgress);
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    execute(apprenticeId);
  }, [apprenticeId, execute]);

  useEffect(() => {
    if (data?.progress) {
      setProgress(data.progress);
    }
  }, [data]);

  if (loading) {
    return <div className="alert alert-info">Loading progress...</div>;
  }

  if (error) {
    return <div className="alert alert-danger">Error: {error}</div>;
  }

  if (!progress) {
    return <div className="alert alert-info">No progress data available</div>;
  }

  const supervisionCompliant = progress.supervisionOk ? '✓ Compliant' : '⚠ Below Minimum';
  const supervisionStyle = progress.supervisionOk ? 'success' : 'warning';

  return (
    <div className="progress-container">
      <div className="progress-section">
        <h3>Overall Completion</h3>
        <div className="progress-bar-large">
          <div
            className="progress-fill"
            style={{ width: `${progress.completionPercent || 0}%` }}
          />
        </div>
        <p className="progress-text">{(progress.completionPercent || 0).toFixed(1)}% Complete</p>
      </div>

      <div className="progress-stats">
        <div className="stat-box">
          <div className="stat-label">OJT Hours Logged</div>
          <div className="stat-number">{(progress.ojtHours || 0).toFixed(1)}</div>
          <div className="stat-sub">hours</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Qualified Hours</div>
          <div className="stat-number">{(progress.qualifiedOjt || 0).toFixed(1)}</div>
          <div className="stat-sub">hours approved</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Supervision Ratio</div>
          <div className="stat-number">{(progress.supervisionRatio || 0).toFixed(2)}%</div>
          <div className={`stat-sub status-${supervisionStyle}`}>{supervisionCompliant}</div>
        </div>
      </div>

      {progress.supervisionRatio !== undefined && (
        <div className={`alert alert-${supervisionStyle}`}>
          <strong>Supervision Ratio:</strong> {(progress.supervisionRatio || 0).toFixed(2)}%
          {progress.supervisionOk
            ? ' ✓ Meets BACB 2.5% minimum requirement'
            : ' ⚠ Does not meet BACB 2.5% minimum requirement'
          }
        </div>
      )}
    </div>
  );
};
