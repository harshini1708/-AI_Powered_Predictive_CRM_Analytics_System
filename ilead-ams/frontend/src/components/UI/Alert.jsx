import React from 'react';
import { useTheme } from '../../context/ThemeContext';

/**
 * Alert Component
 */
export const Alert = ({
  type = 'info',
  title,
  message,
  onClose,
  dismissible = true,
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const getTypeStyles = () => {
    switch (type) {
      case 'success':
        return {
          background: currentTheme.success,
          color: currentTheme.white,
          border: `1px solid ${currentTheme.success}`,
          icon: '✅'
        };
      case 'danger':
        return {
          background: currentTheme.danger,
          color: currentTheme.white,
          border: `1px solid ${currentTheme.danger}`,
          icon: '❌'
        };
      case 'warning':
        return {
          background: currentTheme.warning,
          color: currentTheme.white,
          border: `1px solid ${currentTheme.warning}`,
          icon: '⚠️'
        };
      case 'info':
      default:
        return {
          background: currentTheme.info,
          color: currentTheme.white,
          border: `1px solid ${currentTheme.info}`,
          icon: 'ℹ️'
        };
    }
  };

  const typeStyles = getTypeStyles();

  const alertStyle = {
    padding: '12px 16px',
    borderRadius: '6px',
    border: typeStyles.border,
    background: typeStyles.background,
    color: typeStyles.color,
    display: 'flex',
    alignItems: 'flex-start',
    gap: '12px',
    marginBottom: '16px',
    ...style
  };

  return (
    <div style={alertStyle} {...props}>
      <span style={{ fontSize: '18px', marginTop: '2px' }}>
        {typeStyles.icon}
      </span>
      <div style={{ flex: 1 }}>
        {title && (
          <h4 style={{ margin: '0 0 4px 0', fontSize: '14px', fontWeight: '600' }}>
            {title}
          </h4>
        )}
        {message && (
          <p style={{ margin: 0, fontSize: '14px' }}>
            {message}
          </p>
        )}
      </div>
      {dismissible && (
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'inherit',
            cursor: 'pointer',
            fontSize: '18px',
            padding: '0 4px'
          }}
        >
          ✕
        </button>
      )}
    </div>
  );
};

/**
 * Badge Component
 */
export const Badge = ({
  children,
  variant = 'default',
  size = 'md',
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const variantStyles = {
    default: {
      background: currentTheme.gray300,
      color: currentTheme.text
    },
    primary: {
      background: currentTheme.primary,
      color: currentTheme.white
    },
    success: {
      background: currentTheme.success,
      color: currentTheme.white
    },
    danger: {
      background: currentTheme.danger,
      color: currentTheme.white
    },
    warning: {
      background: currentTheme.warning,
      color: currentTheme.white
    }
  };

  const sizeStyles = {
    sm: { padding: '2px 6px', fontSize: '11px' },
    md: { padding: '4px 8px', fontSize: '12px' },
    lg: { padding: '6px 12px', fontSize: '14px' }
  };

  const badgeStyle = {
    display: 'inline-block',
    borderRadius: '12px',
    fontWeight: '500',
    ...variantStyles[variant],
    ...sizeStyles[size],
    ...style
  };

  return (
    <span style={badgeStyle} {...props}>
      {children}
    </span>
  );
};

/**
 * Tag Component
 */
export const Tag = ({
  children,
  onClose,
  color = 'blue',
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const colors = {
    blue: { bg: currentTheme.primaryLight, text: currentTheme.white },
    green: { bg: currentTheme.success, text: currentTheme.white },
    red: { bg: currentTheme.danger, text: currentTheme.white },
    yellow: { bg: currentTheme.warning, text: currentTheme.white },
    gray: { bg: currentTheme.gray300, text: currentTheme.text }
  };

  const colorStyle = colors[color] || colors.blue;

  const tagStyle = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    borderRadius: '16px',
    background: colorStyle.bg,
    color: colorStyle.text,
    fontSize: '12px',
    fontWeight: '500',
    ...style
  };

  return (
    <span style={tagStyle} {...props}>
      {children}
      {onClose && (
        <button
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'inherit',
            cursor: 'pointer',
            fontSize: '14px',
            padding: 0,
            marginLeft: '4px'
          }}
        >
          ✕
        </button>
      )}
    </span>
  );
};
