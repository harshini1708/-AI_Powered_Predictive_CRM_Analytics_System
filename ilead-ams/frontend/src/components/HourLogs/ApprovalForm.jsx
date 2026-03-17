import React, { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as hourLogApi from '../../api/hourLogs';
import '../styles/Forms.css';

const RUBRIC_SCORES = [
  { score: 1, label: 'Not Yet', color: '#dc3545', desc: 'Accuracy <50%' },
  { score: 2, label: 'Emerging', color: '#ff9800', desc: 'Accuracy 50-69%' },
  { score: 3, label: 'Developing', color: '#ffc107', desc: 'Accuracy 70-89%' },
  { score: 4, label: 'Proficient', color: '#28a745', desc: 'Accuracy 90%+' },
  { score: 5, label: 'Mastery', color: '#007bff', desc: 'Expert level' }
];

const TASK_DOMAINS = ['A', 'B', 'C', 'D', 'E', 'F'];

export const ApprovalForm = ({ logId, onSuccess, onCancel }) => {
  const [score, setScore] = useState(null);
  const [domain, setDomain] = useState('');
  const [task, setTask] = useState('');
  const [supervisionMinutes, setSupervisionMinutes] = useState('');
  const [supervisionType, setSupervisionType] = useState('direct');
  const [notes, setNotes] = useState('');
  const [error, setError] = useState(null);

  const approve = useApi(hourLogApi.approveHourLog);
  const reject = useApi(hourLogApi.rejectHourLog);

  const handleApprove = async (e) => {
    e.preventDefault();
    setError(null);

    if (!score || !domain || !task) {
      setError('Please select score, domain, and task');
      return;
    }

    try {
      await approve.execute(logId, {
        rubricScore: score,
        rubricDomain: domain,
        specificTask: task,
        supervisionMinutes: parseInt(supervisionMinutes) || 0,
        supervisionType: supervisionType
      });
      onSuccess?.();
    } catch (err) {
      setError(err.response?.data?.message || 'Approval failed');
    }
  };

  const handleReject = async (e) => {
    e.preventDefault();
    setError(null);

    if (!notes.trim()) {
      setError('Please provide rejection reason');
      return;
    }

    try {
      await reject.execute(logId, { reason: notes });
      onSuccess?.();
    } catch (err) {
      setError(err.response?.data?.message || 'Rejection failed');
    }
  };

  const selectedScore = RUBRIC_SCORES.find(s => s.score === score);

  return (
    <form className="approval-form">
      {error && <div className="alert alert-danger">{error}</div>}

      <div className="form-section">
        <h3>Competency Rating</h3>
        <div className="rubric-grid">
          {RUBRIC_SCORES.map(item => (
            <button
              key={item.score}
              type="button"
              className={`rubric-option ${score === item.score ? 'selected' : ''}`}
              onClick={() => setScore(item.score)}
              style={{
                borderColor: score === item.score ? item.color : '#e0e0e0',
                backgroundColor: score === item.score ? `${item.color}20` : '#fff'
              }}
            >
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: item.color }}>
                {item.score}
              </div>
              <div style={{ fontSize: '11px', fontWeight: '600' }}>{item.label}</div>
              <div style={{ fontSize: '10px', color: '#666' }}>{item.desc}</div>
            </button>
          ))}
        </div>
        {selectedScore && (
          <div className="alert alert-info">
            Score {selectedScore.score}: {selectedScore.label} - {selectedScore.desc}
          </div>
        )}
      </div>

      <div className="form-section">
        <h3>Task Mapping</h3>
        <div className="form-row">
          <div className="form-group">
            <label>BACB Domain *</label>
            <select
              value={domain}
              onChange={(e) => setDomain(e.target.value)}
              disabled={approve.loading || reject.loading}
            >
              <option value="">Select domain...</option>
              {TASK_DOMAINS.map(d => (
                <option key={d} value={d}>{d} - RBT Competency Domain</option>
              ))}
            </select>
          </div>
          <div className="form-group">
            <label>Specific Task *</label>
            <input
              type="text"
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="e.g., A-1: Professional Conduct"
              disabled={approve.loading || reject.loading}
            />
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>Supervision Tracking</h3>
        <div className="form-row">
          <div className="form-group">
            <label>Supervision Minutes</label>
            <input
              type="number"
              value={supervisionMinutes}
              onChange={(e) => setSupervisionMinutes(e.target.value)}
              placeholder="0"
              min="0"
              disabled={approve.loading || reject.loading}
            />
          </div>
          <div className="form-group">
            <label>Supervision Type</label>
            <select
              value={supervisionType}
              onChange={(e) => setSupervisionType(e.target.value)}
              disabled={approve.loading || reject.loading}
            >
              <option value="direct">Direct Supervision</option>
              <option value="indirect">Indirect Supervision</option>
            </select>
          </div>
        </div>
      </div>

      <div className="form-section">
        <h3>Notes</h3>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Feedback, observations, or rejection reason"
          rows="4"
          disabled={approve.loading || reject.loading}
        />
      </div>

      <div className="form-actions">
        <button
          type="submit"
          className="btn btn-success"
          onClick={handleApprove}
          disabled={approve.loading || reject.loading}
        >
          {approve.loading ? 'Approving...' : '✓ Approve Hours'}
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={handleReject}
          disabled={approve.loading || reject.loading}
        >
          {reject.loading ? 'Rejecting...' : '✗ Reject Hours'}
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={approve.loading || reject.loading}
        >
          Cancel
        </button>
      </div>
    </form>
  );
};
