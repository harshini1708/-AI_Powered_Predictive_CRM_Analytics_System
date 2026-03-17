import React, { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import '../styles/Layout.css';

export const Navbar = () => {
  const { user, logout } = useAuth();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const getRoleLabel = (role) => {
    const labels = {
      apprentice: 'Apprentice',
      supervisor: 'Supervisor',
      journeyworker: 'Journeyworker',
      administrator: 'Administrator',
      super_admin: 'Super Admin'
    };
    return labels[role] || role;
  };

  const handleLogout = async () => {
    await logout();
    setShowUserMenu(false);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <h1 className="logo-text">I-LEAD AMS</h1>
          <span className="tier-badge">AMS</span>
        </div>

        {user && (
          <div className="navbar-user">
            <div className="user-info">
              <span className="user-name">{user.firstName} {user.lastName}</span>
              <span className="user-role">{getRoleLabel(user.role)}</span>
            </div>

            <div className="user-menu-container">
              <button
                className="user-avatar-btn"
                onClick={() => setShowUserMenu(!showUserMenu)}
                title={`${user.firstName} ${user.lastName}`}
              >
                {user.firstName.charAt(0).toUpperCase()}
              </button>

              {showUserMenu && (
                <div className="user-dropdown">
                  <div className="dropdown-item">
                    <strong>{user.email}</strong>
                  </div>
                  <div className="dropdown-divider"></div>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      // Navigate to profile
                      setShowUserMenu(false);
                    }}
                  >
                    Profile Settings
                  </button>
                  <button
                    className="dropdown-item"
                    onClick={() => {
                      // Navigate to preferences
                      setShowUserMenu(false);
                    }}
                  >
                    Preferences
                  </button>
                  <div className="dropdown-divider"></div>
                  <button
                    className="dropdown-item logout"
                    onClick={handleLogout}
                  >
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
