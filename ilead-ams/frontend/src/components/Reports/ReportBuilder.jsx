import React, { useState, useEffect } from 'react';
import { useApi } from '../../hooks/useApi';
import * as reportApi from '../../api/reports';

export const ReportBuilder = ({ onSave, initialReport = null }) => {
  const { data: templates, execute: loadTemplates } = useApi(reportApi.getReportTemplates);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    type: 'hours_summary',
    filters: {
      dateRange: {
        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        endDate: new Date().toISOString().split('T')[0]
      },
      apprenticeIds: [],
      supervisorIds: [],
      programs: [],
      status: []
    },
    columns: [],
    groupBy: { field: 'none' },
    sortBy: { field: 'date', order: 'desc' },
    displayOptions: {
      showChart: true,
      chartType: 'bar',
      pageSize: 50,
      showTotals: true,
      showPagination: true
    },
    schedule: {
      enabled: false,
      frequency: 'weekly',
      time: '09:00'
    },
    delivery: {
      enabled: false,
      method: 'email',
      recipients: []
    }
  });

  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadTemplates();
    if (initialReport) {
      setFormData(initialReport);
    }
  }, [loadTemplates, initialReport]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleDateChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      filters: {
        ...prev.filters,
        dateRange: {
          ...prev.filters.dateRange,
          [field]: value
        }
      }
    }));
  };

  const handleScheduleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      schedule: {
        ...prev.schedule,
        [field]: value
      }
    }));
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      await onSave(formData);
      alert('Report saved successfully!');
    } catch (error) {
      alert('Error saving report: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const renderStep1 = () => (
    <div style={{ padding: '20px' }}>
      <h3>Report Details</h3>
      <div style={{ marginBottom: '15px' }}>
        <label>Report Name *</label>
        <input
          type="text"
          value={formData.name}
          onChange={(e) => handleInputChange('name', e.target.value)}
          placeholder="e.g., Weekly Hours Summary"
          style={{ width: '100%', padding: '8px', marginTop: '5px' }}
        />
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label>Description</label>
        <textarea
          value={formData.description}
          onChange={(e) => handleInputChange('description', e.target.value)}
          placeholder="Optional description"
          style={{ width: '100%', padding: '8px', marginTop: '5px', minHeight: '80px' }}
        />
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label>Report Type</label>
        <select
          value={formData.type}
          onChange={(e) => handleInputChange('type', e.target.value)}
          style={{ width: '100%', padding: '8px', marginTop: '5px' }}
        >
          {templates && templates.map(t => (
            <option key={t.id} value={t.id}>{t.name}</option>
          ))}
        </select>
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div style={{ padding: '20px' }}>
      <h3>Date Range & Filters</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
        <div>
          <label>Start Date</label>
          <input
            type="date"
            value={formData.filters.dateRange.startDate}
            onChange={(e) => handleDateChange('startDate', e.target.value)}
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          />
        </div>
        <div>
          <label>End Date</label>
          <input
            type="date"
            value={formData.filters.dateRange.endDate}
            onChange={(e) => handleDateChange('endDate', e.target.value)}
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          />
        </div>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label>Status Filter</label>
        <div style={{ marginTop: '5px' }}>
          {['pending', 'approved', 'rejected'].map(status => (
            <label key={status} style={{ marginRight: '15px' }}>
              <input
                type="checkbox"
                checked={formData.filters.status.includes(status)}
                onChange={(e) => {
                  const statuses = e.target.checked
                    ? [...formData.filters.status, status]
                    : formData.filters.status.filter(s => s !== status);
                  setFormData(prev => ({
                    ...prev,
                    filters: { ...prev.filters, status: statuses }
                  }));
                }}
              />
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </label>
          ))}
        </div>
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div style={{ padding: '20px' }}>
      <h3>Display Options</h3>
      <div style={{ marginBottom: '15px' }}>
        <label>
          <input
            type="checkbox"
            checked={formData.displayOptions.showChart}
            onChange={(e) => setFormData(prev => ({
              ...prev,
              displayOptions: { ...prev.displayOptions, showChart: e.target.checked }
            }))}
          />
          Show Chart
        </label>
      </div>

      {formData.displayOptions.showChart && (
        <div style={{ marginBottom: '15px' }}>
          <label>Chart Type</label>
          <select
            value={formData.displayOptions.chartType}
            onChange={(e) => setFormData(prev => ({
              ...prev,
              displayOptions: { ...prev.displayOptions, chartType: e.target.value }
            }))}
            style={{ width: '100%', padding: '8px', marginTop: '5px' }}
          >
            <option value="bar">Bar Chart</option>
            <option value="line">Line Chart</option>
            <option value="pie">Pie Chart</option>
            <option value="area">Area Chart</option>
          </select>
        </div>
      )}

      <div style={{ marginBottom: '15px' }}>
        <label>
          <input
            type="checkbox"
            checked={formData.displayOptions.showTotals}
            onChange={(e) => setFormData(prev => ({
              ...prev,
              displayOptions: { ...prev.displayOptions, showTotals: e.target.checked }
            }))}
          />
          Show Totals
        </label>
      </div>

      <div style={{ marginBottom: '15px' }}>
        <label>Page Size</label>
        <input
          type="number"
          value={formData.displayOptions.pageSize}
          onChange={(e) => setFormData(prev => ({
            ...prev,
            displayOptions: { ...prev.displayOptions, pageSize: parseInt(e.target.value) }
          }))}
          style={{ width: '100%', padding: '8px', marginTop: '5px' }}
        />
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div style={{ padding: '20px' }}>
      <h3>Scheduling & Delivery</h3>
      
      <div style={{ marginBottom: '15px' }}>
        <label>
          <input
            type="checkbox"
            checked={formData.schedule.enabled}
            onChange={(e) => handleScheduleChange('enabled', e.target.checked)}
          />
          Schedule Report
        </label>
      </div>

      {formData.schedule.enabled && (
        <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '5px', marginBottom: '15px' }}>
          <div style={{ marginBottom: '10px' }}>
            <label>Frequency</label>
            <select
              value={formData.schedule.frequency}
              onChange={(e) => handleScheduleChange('frequency', e.target.value)}
              style={{ width: '100%', padding: '8px', marginTop: '5px' }}
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
              <option value="quarterly">Quarterly</option>
              <option value="yearly">Yearly</option>
            </select>
          </div>

          <div>
            <label>Time</label>
            <input
              type="time"
              value={formData.schedule.time}
              onChange={(e) => handleScheduleChange('time', e.target.value)}
              style={{ width: '100%', padding: '8px', marginTop: '5px' }}
            />
          </div>
        </div>
      )}

      <div style={{ marginBottom: '15px' }}>
        <label>
          <input
            type="checkbox"
            checked={formData.delivery.enabled}
            onChange={(e) => setFormData(prev => ({
              ...prev,
              delivery: { ...prev.delivery, enabled: e.target.checked }
            }))}
          />
          Enable Email Delivery
        </label>
      </div>

      {formData.delivery.enabled && (
        <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '5px' }}>
          <label>Recipients (comma-separated emails)</label>
          <textarea
            value={formData.delivery.recipients.join(', ')}
            onChange={(e) => setFormData(prev => ({
              ...prev,
              delivery: {
                ...prev.delivery,
                recipients: e.target.value.split(',').map(e => e.trim()).filter(Boolean)
              }
            }))}
            placeholder="user1@example.com, user2@example.com"
            style={{ width: '100%', padding: '8px', marginTop: '5px', minHeight: '60px' }}
          />
        </div>
      )}
    </div>
  );

  return (
    <div style={{
      background: 'white',
      borderRadius: '8px',
      border: '1px solid #ddd',
      overflow: 'hidden'
    }}>
      {/* Steps */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid #eee',
        background: '#f8f9fa'
      }}>
        {[1, 2, 3, 4].map(s => (
          <button
            key={s}
            onClick={() => setStep(s)}
            style={{
              flex: 1,
              padding: '15px',
              border: 'none',
              background: step === s ? '#003d82' : 'transparent',
              color: step === s ? 'white' : '#333',
              cursor: 'pointer',
              borderRadius: 0,
              fontWeight: step === s ? 'bold' : 'normal'
            }}
          >
            {['Details', 'Filters', 'Display', 'Schedule'][s - 1]}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ minHeight: '300px' }}>
        {step === 1 && renderStep1()}
        {step === 2 && renderStep2()}
        {step === 3 && renderStep3()}
        {step === 4 && renderStep4()}
      </div>

      {/* Navigation */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        padding: '20px',
        borderTop: '1px solid #eee',
        background: '#f8f9fa'
      }}>
        <button
          onClick={() => setStep(Math.max(1, step - 1))}
          disabled={step === 1}
          style={{
            padding: '10px 20px',
            background: '#6c757d',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: step === 1 ? 'not-allowed' : 'pointer',
            opacity: step === 1 ? 0.5 : 1
          }}
        >
          Previous
        </button>

        {step < 4 ? (
          <button
            onClick={() => setStep(step + 1)}
            style={{
              padding: '10px 20px',
              background: '#003d82',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Next
          </button>
        ) : (
          <button
            onClick={handleSave}
            disabled={loading || !formData.name}
            style={{
              padding: '10px 20px',
              background: '#28a745',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Saving...' : 'Save Report'}
          </button>
        )}
      </div>
    </div>
  );
};
