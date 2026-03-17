import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as userApi from '../../api/users';
import '../styles/Tables.css';

export const UserList = ({ onSelectUser, onRefresh, onDataChange }) => {
  const { data, loading, error, execute } = useApi(userApi.listUsers);
  const [users, setUsers] = useState([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    execute({
      status: filter !== 'all' ? filter : undefined
    });
  }, [filter, execute, onRefresh]);

  useEffect(() => {
    if (data?.data) {
      setUsers(data.data);
      onDataChange?.(data.data);
    }
  }, [data, onDataChange]);

  const getRoleColor = (role) => {
    const colors = {
      apprentice: '#e84c1f',
      supervisor: '#ff9800',
      journeyworker: '#28a745',
      administrator: '#003d82',
      super_admin: '#6f42c1'
    };
    return colors[role] || '#6c757d';
  };

  if (loading) {
    return <div className="alert alert-info">Loading users...</div>;
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
            <option value="pending_approval">Pending Approval</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>
      </div>

      {users.length === 0 ? (
        <div className="alert alert-info">No users found</div>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Email</th>
              <th>Role</th>
              <th>Status</th>
              <th>Joined</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {users.map(user => (
              <tr key={user.id}>
                <td>
                  <strong>{user.firstName} {user.lastName}</strong>
                </td>
                <td>{user.email}</td>
                <td>
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '4px 8px',
                      backgroundColor: getRoleColor(user.role) + '20',
                      color: getRoleColor(user.role),
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: '600',
                      textTransform: 'capitalize'
                    }}
                  >
                    {user.role.replace('_', ' ')}
                  </span>
                </td>
                <td>
                  <span className={`badge badge-${user.status.toLowerCase()}`}>
                    {user.status.charAt(0).toUpperCase() + user.status.slice(1).replace('_', ' ')}
                  </span>
                </td>
                <td>{new Date(user.createdAt).toLocaleDateString()}</td>
                <td>
                  <button
                    className="btn-sm btn-link"
                    onClick={() => onSelectUser?.(user.id)}
                  >
                    Manage
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
