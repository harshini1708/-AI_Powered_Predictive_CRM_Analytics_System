import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as apprenticeApi from '../../api/apprentices';
import '../styles/Tables.css';

export const ApprenticeList = ({ onSelectApprentice, onDataChange }) => {
  const { data, loading, error, execute } = useApi(apprenticeApi.listApprentices);
  const [apprentices, setApprentices] = useState([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    execute({ status: filter !== 'all' ? filter : undefined });
  }, [filter, execute]);

  useEffect(() => {
    if (data?.data) {
      setApprentices(data.data);
      onDataChange?.(data.data);
    }
  }, [data, onDataChange]);

  if (loading) {
    return <div className="alert alert-info">Loading apprentices...</div>;
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
            <option value="active">Active</option>
            <option value="completed">Completed</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>

      {apprentices.length === 0 ? (
        <div className="alert alert-info">No apprentices found</div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Email</th>
              <th>Program</th>
              <th>Status</th>
              <th>OJT Hours</th>
              <th>Qualified Hours</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {apprentices.map(app => (
              <tr key={app.id}>
                <td>
                  <strong>{app.name}</strong>
                </td>
                <td>{app.email}</td>
                <td>{app.program}</td>
                <td>
                  <span className={`badge badge-${app.status.toLowerCase()}`}>
                    {app.status}
                  </span>
                </td>
                <td>{app.ojtHours || 0}</td>
                <td>{app.qualifiedOjtHours || 0}</td>
                <td>
                  <button
                    className="btn-sm btn-link"
                    onClick={() => onSelectApprentice?.(app.id)}
                  >
                    View
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
