import React, { useEffect, useState } from 'react';
import { useApi } from '../../hooks/useApi';
import * as notificationApi from '../../api/notifications';

export const NotificationBell = () => {
  const { data, execute } = useApi(notificationApi.getUnreadCount);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showDropdown, setShowDropdown] = useState(false);

  useEffect(() => {
    execute();
    
    // Poll for new notifications every 30 seconds
    const interval = setInterval(() => {
      execute();
    }, 30000);

    return () => clearInterval(interval);
  }, [execute]);

  useEffect(() => {
    if (data?.unreadCount !== undefined) {
      setUnreadCount(data.unreadCount);
    }
  }, [data]);

  return (
    <div className="notification-bell" style={{ position: 'relative' }}>
      <button
        className="bell-button"
        onClick={() => setShowDropdown(!showDropdown)}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          fontSize: '1.5em',
          position: 'relative'
        }}
      >
        🔔
        {unreadCount > 0 && (
          <span className="notification-badge" style={{
            position: 'absolute',
            top: '-5px',
            right: '-8px',
            background: '#e84c1f',
            color: 'white',
            borderRadius: '50%',
            width: '22px',
            height: '22px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '11px',
            fontWeight: 'bold'
          }}>
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </button>

      {showDropdown && (
        <div style={{
          position: 'absolute',
          right: 0,
          top: '40px',
          background: 'white',
          border: '1px solid #ddd',
          borderRadius: '8px',
          width: '320px',
          maxHeight: '400px',
          overflowY: 'auto',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          zIndex: 1000
        }}>
          <div style={{ padding: '15px', borderBottom: '1px solid #eee', fontWeight: 'bold' }}>
            Notifications
          </div>
          <div style={{ padding: '15px', textAlign: 'center', color: '#666' }}>
            Notification center coming soon...
          </div>
        </div>
      )}
    </div>
  );
};
