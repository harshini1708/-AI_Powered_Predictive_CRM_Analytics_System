import React from 'react';
import { useTheme } from '../../context/ThemeContext';

/**
 * Reusable Button Component with multiple variants
 */
export const Button = ({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  className = '',
  style = {},
  type = 'button',
  ...props
}) => {
  const { currentTheme } = useTheme();

  const getVariantStyles = () => {
    const baseStyle = {
      padding: size === 'sm' ? '8px 12px' : size === 'lg' ? '12px 24px' : '10px 16px',
      fontSize: size === 'sm' ? '12px' : size === 'lg' ? '16px' : '14px',
      fontWeight: '500',
      border: 'none',
      borderRadius: '6px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      transition: 'all 0.2s ease',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      opacity: disabled ? 0.6 : 1,
      ...style
    };

    switch (variant) {
      case 'primary':
        return {
          ...baseStyle,
          background: currentTheme.primary,
          color: currentTheme.white,
          boxShadow: currentTheme.shadowSm,
        };
      case 'secondary':
        return {
          ...baseStyle,
          background: currentTheme.secondary,
          color: currentTheme.white,
          boxShadow: currentTheme.shadowSm,
        };
      case 'success':
        return {
          ...baseStyle,
          background: currentTheme.success,
          color: currentTheme.white,
        };
      case 'danger':
        return {
          ...baseStyle,
          background: currentTheme.danger,
          color: currentTheme.white,
        };
      case 'outline':
        return {
          ...baseStyle,
          background: 'transparent',
          color: currentTheme.primary,
          border: `2px solid ${currentTheme.primary}`,
        };
      case 'ghost':
        return {
          ...baseStyle,
          background: 'transparent',
          color: currentTheme.text,
        };
      default:
        return baseStyle;
    }
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      style={getVariantStyles()}
      className={className}
      {...props}
    >
      {loading && <span>⏳</span>}
      {children}
    </button>
  );
};
