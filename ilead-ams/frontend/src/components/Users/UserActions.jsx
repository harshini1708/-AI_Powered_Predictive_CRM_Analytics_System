import React, { useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as userApi from '../../api/users';

export const UserActions = ({ userId, user, onSuccess, onCancel }) => {
  const [action, setAction] = useState(null);
  const approve = useApi(userApi.approveUser);
  const deactivate = useApi(userApi.deactivateUser);

  const handleApprove = async () => {
    try {
      await approve.execute(userId);
      onSuccess?.();
    } catch (err) {
      console.error('Approval failed:', err);
    }
  };

  const handleDeactivate = async () => {
    if (window.confirm(`Are you sure you want to deactivate ${user?.firstName} ${user?.lastName}?`)) {
      try {
        await deactivate.execute(userId);
        onSuccess?.();
      } catch (err) {
        console.error('Deactivation failed:', err);
      }
    }
  };

  if (!user) return null;

  return (
    <div>
      <div className="alert alert-info">
        <strong>{user.firstName} {user.lastName}</strong><br/>
        Email: {user.email}<br/>
        Role: {user.role.replace('_', ' ')}<br/>
        Status: {user.status}
      </div>

      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        {user.status === 'pending_approval' && (
          <button
            className="btn btn-success"
            onClick={handleApprove}
            disabled={approve.loading}
          >
            {approve.loading ? 'Approving...' : '✓ Approve User'}
          </button>
        )}

        {user.status === 'active' && (
          <button
            className="btn btn-danger"
            onClick={handleDeactivate}
            disabled={deactivate.loading}
          >
            {deactivate.loading ? 'Deactivating...' : '× Deactivate'}
          </button>
        )}

        <button
          className="btn btn-secondary"
          onClick={onCancel}
          disabled={approve.loading || deactivate.loading}
        >
          Close
        </button>
      </div>
    </div>
  );
};
