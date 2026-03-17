import React, { useEffect, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { useApi } from '../../hooks/useApi';
import * as hourLogApi from '../../api/hourLogs';
import '../styles/Tables.css';

export const HourLogList = ({ onSelectLog, apprenticeId, advancedFilters, onLogsChange }) => {
  const { user } = useAuth();
  const { data, loading, error, execute } = useApi(hourLogApi.listHourLogs);
  const [logs, setLogs] = useState([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    const filterParams = {
      status: filter !== 'all' ? filter : undefined,
      apprenticeId: apprenticeId
    };

    // Merge advanced filters if provided
    if (advancedFilters) {
      if (advancedFilters.status && advancedFilters.status !== 'all') {
        filterParams.status = advancedFilters.status;
      }
      if (advancedFilters.dateFrom) {
        filterParams.dateFrom = advancedFilters.dateFrom;
      }
      if (advancedFilters.dateTo) {
        filterParams.dateTo = advancedFilters.dateTo;
      }
      if (advancedFilters.domain && advancedFilters.domain !== 'all') {
        filterParams.domain = advancedFilters.domain;
      }
      if (advancedFilters.minHours) {
        filterParams.minHours = advancedFilters.minHours;
      }
      if (advancedFilters.maxHours) {
        filterParams.maxHours = advancedFilters.maxHours;
      }
      if (advancedFilters.sortBy) {
        filterParams.sortBy = advancedFilters.sortBy;
      }
    }

    execute(filterParams);
  }, [filter, apprenticeId, advancedFilters, execute]);

  useEffect(() => {
    if (data?.data) {
      setLogs(data.data);
      onLogsChange?.(data.data);
    }
  }, [data, onLogsChange]);

  const getStatusBadgeClass = (status) => {
    const statusMap = {
      pending: 'pending',
      approved: 'approved',
      rejected: 'rejected'
    };
    return statusMap[status] || status;
  };

  if (loading) {
    return <div className="alert alert-info">Loading hour logs...</div>;
  }

  if (error) {
    return <div className="alert alert-danger">Error: {error}</div>;
  }

  return (
    <div>
      <div className="table-toolbar">
        <div className="filter-group">
          <label>Filter by Status:</label>
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>
      </div>

      {logs.length === 0 ? (
        <div className="alert alert-info">No hour logs found</div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th>
              <th>Apprentice</th>
              <th>Domain/Category</th>
              <th>OJT Hours</th>
              <th>RTI Hours</th>
              <th>Qualified</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {logs.map(log => (
              <tr key={log.id}>
                <td>{new Date(log.date).toLocaleDateString()}</td>
                <td>{log.apprenticeName || log.apprenticeId}</td>
                <td>{log.ojtDomainName || log.rtiCategoryName || '—'}</td>
                <td>{log.ojtHours || 0}</td>
                <td>{log.rtiHours || 0}</td>
                <td>{log.qualifiedHours || 0}</td>
                <td>
                  <span className={`badge badge-${getStatusBadgeClass(log.status)}`}>
                    {log.status.charAt(0).toUpperCase() + log.status.slice(1)}
                  </span>
                </td>
                <td>
                  <button
                    className="btn-sm btn-link"
                    onClick={() => onSelectLog?.(log.id)}
                  >
                    {log.status === 'pending' ? 'Review' : 'View'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
