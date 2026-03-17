import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import * as hourLogApi from '../api/hourLogs';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';

const OJT_DOMAINS = [
  { key: 'dataCollection', label: 'Data Collection' },
  { key: 'assessment', label: 'Assessment Support' },
  { key: 'skillAcquisition', label: 'Skill Acquisition' },
  { key: 'behaviorReduction', label: 'Behavior Reduction' },
  { key: 'documentation', label: 'Documentation & Compliance' },
  { key: 'crisisManagement', label: 'Crisis Management' }
];

export const SubmitHoursPage = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState(null);
  const [formData, setFormData] = useState({
    ojtDomain: '',
    ojtHours: '',
    rtiHours: '',
    logDate: new Date().toISOString().split('T')[0],
    description: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    if (!formData.ojtDomain && !formData.ojtHours && !formData.rtiHours) {
      setError('Please select a domain and enter hours');
      return;
    }

    try {
      setLoading(true);
      await hourLogApi.submitHours({
        ojtDomain: formData.ojtDomain || null,
        ojtHours: parseFloat(formData.ojtHours) || 0,
        rtiHours: parseFloat(formData.rtiHours) || 0,
        logDate: formData.logDate,
        description: formData.description
      });
      setSuccess(true);
      setFormData({
        ojtDomain: '',
        ojtHours: '',
        rtiHours: '',
        logDate: new Date().toISOString().split('T')[0],
        description: ''
      });
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to submit hours');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/hours/submit" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Submit Hours</h1>
              <p className="page-subtitle">Log your apprenticeship hours</p>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Hour Log Submission</div>

            {success && (
              <div className="alert alert-success">
                ✓ Hours submitted successfully for approval!
              </div>
            )}
            {error && (
              <div className="alert alert-danger">{error}</div>
            )}

            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Date *</label>
                <input
                  type="date"
                  name="logDate"
                  value={formData.logDate}
                  onChange={handleChange}
                  required
                  disabled={loading}
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>OJT Domain</label>
                  <select
                    name="ojtDomain"
                    value={formData.ojtDomain}
                    onChange={handleChange}
                    disabled={loading}
                  >
                    <option value="">Select domain...</option>
                    {OJT_DOMAINS.map(domain => (
                      <option key={domain.key} value={domain.key}>
                        {domain.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label>OJT Hours</label>
                  <input
                    type="number"
                    name="ojtHours"
                    value={formData.ojtHours}
                    onChange={handleChange}
                    placeholder="0"
                    step="0.5"
                    min="0"
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>RTI Hours</label>
                <input
                  type="number"
                  name="rtiHours"
                  value={formData.rtiHours}
                  onChange={handleChange}
                  placeholder="0"
                  step="0.5"
                  min="0"
                  disabled={loading}
                />
              </div>

              <div className="form-group">
                <label>Description / Notes</label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  placeholder="What did you work on?"
                  rows="4"
                  disabled={loading}
                />
              </div>

              <button
                type="submit"
                className="btn"
                disabled={loading}
              >
                {loading ? 'Submitting...' : 'Submit Hours for Approval'}
              </button>
            </form>
          </div>

          <div className="card" style={{ marginTop: '24px' }}>
            <div className="card-title">How to Submit Hours</div>
            <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
              <li>Select the date you performed the work</li>
              <li>Choose the OJT domain (if applicable)</li>
              <li>Enter the hours worked (OJT and/or RTI)</li>
              <li>Provide a brief description of what you accomplished</li>
              <li>Click "Submit Hours for Approval"</li>
              <li>Your supervisor will review and approve the hours</li>
              <li>Approved hours count toward your apprenticeship requirements</li>
            </ul>
          </div>
        </main>
      </div>
    </div>
  );
};
