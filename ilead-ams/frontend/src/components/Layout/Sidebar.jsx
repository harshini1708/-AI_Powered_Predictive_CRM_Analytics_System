import React from 'react';
import { useAuth } from '../../hooks/useAuth';
import '../styles/Layout.css';

const MENU_ITEMS = {
  apprentice: [
    { label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { label: 'My Progress', path: '/progress', icon: '📈' },
    { label: 'Submit Hours', path: '/hours/submit', icon: '⏱️' },
    { label: 'Hour Logs', path: '/hours', icon: '📋' }
  ],
  supervisor: [
    { label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { label: 'Apprentices', path: '/apprentices', icon: '👥' },
    { label: 'Hour Approvals', path: '/approvals', icon: '✅' },
    { label: 'Reports', path: '/reports', icon: '📄' }
  ],
  journeyworker: [
    { label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { label: 'Apprentices', path: '/apprentices', icon: '👥' },
    { label: 'Hour Approvals', path: '/approvals', icon: '✅' },
    { label: 'Reports', path: '/reports', icon: '📄' }
  ],
  administrator: [
    { label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { label: 'Users', path: '/users', icon: '👤' },
    { label: 'Apprentices', path: '/apprentices', icon: '👥' },
    { label: 'Programs', path: '/programs', icon: '🎓' },
    { label: 'Hour Approvals', path: '/approvals', icon: '✅' },
    { label: 'Analytics', path: '/analytics', icon: '📈' },
    { label: 'Documents', path: '/documents', icon: '📁' },
    { label: 'Reports', path: '/reports', icon: '📄' }
  ],
  super_admin: [
    { label: 'Dashboard', path: '/dashboard', icon: '📊' },
    { label: 'Users', path: '/users', icon: '👤' },
    { label: 'Apprentices', path: '/apprentices', icon: '👥' },
    { label: 'Programs', path: '/programs', icon: '🎓' },
    { label: 'Organizations', path: '/organizations', icon: '🏢' },
    { label: 'Analytics', path: '/analytics', icon: '📈' },
    { label: 'Documents', path: '/documents', icon: '📁' },
    { label: 'Approvals', path: '/approvals', icon: '✅' },
    { label: 'Reports', path: '/reports', icon: '📄' },
    { label: 'System', path: '/system', icon: '⚙️' }
  ]
};

export const Sidebar = ({ activeRoute = '/' }) => {
  const { user } = useAuth();

  if (!user) return null;

  const isSuperAdmin = user.role === 'super_admin';
  const menuItems = MENU_ITEMS[user.role] || MENU_ITEMS.apprentice;

  return (
    <aside className={`sidebar ${isSuperAdmin ? 'sidebar-super' : ''}`}>
      <div className="sidebar-section">
        <div className="sidebar-section-title">Main Menu</div>
        {menuItems.map((item) => (
          <a
            key={item.path}
            href={item.path}
            className={`sidebar-item ${activeRoute === item.path ? 'active' : ''}`}
            onClick={(e) => {
              // In a real app, this would use React Router
              // e.preventDefault();
              // navigate(item.path);
            }}
          >
            <span className="sidebar-item-icon">{item.icon}</span>
            <span>{item.label}</span>
          </a>
        ))}
      </div>

      {isSuperAdmin && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-section-title">Administration</div>
            <a href="/admin/settings" className="sidebar-item">
              <span className="sidebar-item-icon">⚙️</span>
              <span>Settings</span>
            </a>
            <a href="/admin/audit" className="sidebar-item">
              <span className="sidebar-item-icon">🔍</span>
              <span>Audit Logs</span>
            </a>
          </div>
        </>
      )}
    </aside>
  );
};
