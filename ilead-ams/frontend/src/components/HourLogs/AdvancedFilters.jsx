import React, { useState } from 'react';
import '../styles/Filters.css';

export const AdvancedFilters = ({ onApplyFilters, onReset }) => {
  const [filters, setFilters] = useState({
    status: 'all',
    dateFrom: '',
    dateTo: '',
    domain: 'all',
    minHours: '',
    maxHours: '',
    apprenticeId: '',
    sortBy: 'date_desc'
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleApply = () => {
    onApplyFilters?.(filters);
  };

  const handleReset = () => {
    setFilters({
      status: 'all',
      dateFrom: '',
      dateTo: '',
      domain: 'all',
      minHours: '',
      maxHours: '',
      apprenticeId: '',
      sortBy: 'date_desc'
    });
    onReset?.();
  };

  const activeFilterCount = Object.values(filters).filter(
    v => v && v !== 'all' && v !== 'date_desc'
  ).length;

  return (
    <div className="advanced-filters">
      <div className="filter-header">
        <h3>Filters {activeFilterCount > 0 && <span className="filter-badge">{activeFilterCount}</span>}</h3>
        <button
          className="toggle-btn"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          {showAdvanced ? '△ Hide' : '▽ Show'} Advanced
        </button>
      </div>

      <div className="basic-filters">
        <div className="filter-group">
          <label>Status</label>
          <select name="status" value={filters.status} onChange={handleChange}>
            <option value="all">All</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Sort By</label>
          <select name="sortBy" value={filters.sortBy} onChange={handleChange}>
            <option value="date_desc">Newest First</option>
            <option value="date_asc">Oldest First</option>
            <option value="hours_desc">Most Hours</option>
            <option value="hours_asc">Least Hours</option>
          </select>
        </div>
      </div>

      {showAdvanced && (
        <div className="advanced-filter-section">
          <h4>Advanced Options</h4>

          <div className="filter-grid">
            <div className="filter-group">
              <label>Date From</label>
              <input
                type="date"
                name="dateFrom"
                value={filters.dateFrom}
                onChange={handleChange}
              />
            </div>

            <div className="filter-group">
              <label>Date To</label>
              <input
                type="date"
                name="dateTo"
                value={filters.dateTo}
                onChange={handleChange}
              />
            </div>

            <div className="filter-group">
              <label>OJT Domain</label>
              <select name="domain" value={filters.domain} onChange={handleChange}>
                <option value="all">All Domains</option>
                <option value="dataCollection">Data Collection</option>
                <option value="assessment">Assessment</option>
                <option value="skillAcquisition">Skill Acquisition</option>
                <option value="behaviorReduction">Behavior Reduction</option>
                <option value="documentation">Documentation</option>
                <option value="crisisManagement">Crisis Management</option>
              </select>
            </div>

            <div className="filter-group">
              <label>Min Hours</label>
              <input
                type="number"
                name="minHours"
                value={filters.minHours}
                onChange={handleChange}
                placeholder="0"
                min="0"
                step="0.5"
              />
            </div>

            <div className="filter-group">
              <label>Max Hours</label>
              <input
                type="number"
                name="maxHours"
                value={filters.maxHours}
                onChange={handleChange}
                placeholder="100"
                min="0"
                step="0.5"
              />
            </div>

            <div className="filter-group">
              <label>Apprentice ID</label>
              <input
                type="text"
                name="apprenticeId"
                value={filters.apprenticeId}
                onChange={handleChange}
                placeholder="Search apprentice..."
              />
            </div>
          </div>
        </div>
      )}

      <div className="filter-actions">
        <button className="btn btn-primary" onClick={handleApply}>
          ↻ Apply Filters
        </button>
        <button className="btn btn-secondary" onClick={handleReset}>
          Clear All
        </button>
      </div>
    </div>
  );
};
