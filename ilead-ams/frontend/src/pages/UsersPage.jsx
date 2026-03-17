import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { Navbar } from '../components/Layout/Navbar';
import { Sidebar } from '../components/Layout/Sidebar';
import { UserList } from '../components/Users/UserList';
import { UserActions } from '../components/Users/UserActions';
import { exportUsers } from '../utils/export';

export const UsersPage = () => {
  const { user } = useAuth();
  const [selectedUserId, setSelectedUserId] = useState(null);
  const [selectedUser, setSelectedUser] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [users, setUsers] = useState([]);
  const canManage = ['administrator', 'super_admin'].includes(user?.role);

  const handleExport = () => {
    if (users.length === 0) {
      alert('No users to export');
      return;
    }
    exportUsers(users);
  };

  const handleSelectUser = (userId) => {
    // In a real app, fetch user details
    // For now, we'll just get it from the list
    setSelectedUserId(userId);
  };

  const handleActionSuccess = () => {
    setSelectedUserId(null);
    setSelectedUser(null);
    setRefreshTrigger(prev => prev + 1);
  };

  if (!canManage) {
    return (
      <div className="app-main">
        <Navbar />
        <div className="main-layout">
          <Sidebar activeRoute="/users" />
          <main className="main-content">
            <div className="alert alert-danger">
              You don't have permission to manage users.
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div className="app-main">
      <Navbar />
      <div className="main-layout">
        <Sidebar activeRoute="/users" />
        <main className="main-content">
          <div className="page-header">
            <div className="page-header-left">
              <h1 className="page-title">Users</h1>
              <p className="page-subtitle">Manage system users and roles</p>
            </div>
            {!selectedUserId && (
              <div className="page-header-right">
                <button className="btn btn-secondary" onClick={handleExport}>
                  ↓ Export CSV
                </button>
              </div>
            )}
          </div>

          {selectedUserId ? (
            <div className="card">
              <div style={{ marginBottom: '20px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setSelectedUserId(null)}
                >
                  ← Back to List
                </button>
              </div>
              <UserActions
                userId={selectedUserId}
                user={selectedUser}
                onSuccess={handleActionSuccess}
                onCancel={() => setSelectedUserId(null)}
              />
            </div>
          ) : (
            <div className="card">
              <div className="card-title">User Management</div>
              <UserList
                onSelectUser={handleSelectUser}
                onRefresh={refreshTrigger}
                onDataChange={setUsers}
                key={refreshTrigger}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
};
