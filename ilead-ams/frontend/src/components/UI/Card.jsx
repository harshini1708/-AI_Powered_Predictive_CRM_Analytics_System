import React from 'react';
import { useTheme } from '../../context/ThemeContext';

/**
 * Card Component - Container for content
 */
export const Card = ({
  children,
  title,
  subtitle,
  padding = 'md',
  hover = true,
  style = {},
  className = '',
  ...props
}) => {
  const { currentTheme } = useTheme();

  const paddingValues = {
    sm: '12px',
    md: '20px',
    lg: '28px'
  };

  const cardStyle = {
    background: currentTheme.background,
    border: `1px solid ${currentTheme.border}`,
    borderRadius: '8px',
    padding: paddingValues[padding] || paddingValues.md,
    boxShadow: currentTheme.shadowSm,
    transition: hover ? 'all 0.3s ease' : 'none',
    cursor: hover ? 'pointer' : 'default',
    ...style
  };

  return (
    <div style={cardStyle} className={className} {...props}>
      {title && (
        <div style={{ marginBottom: '12px' }}>
          <h3 style={{
            margin: '0 0 4px 0',
            color: currentTheme.text,
            fontSize: '16px',
            fontWeight: '600'
          }}>
            {title}
          </h3>
          {subtitle && (
            <p style={{
              margin: 0,
              color: currentTheme.textSecondary,
              fontSize: '12px'
            }}>
              {subtitle}
            </p>
          )}
        </div>
      )}
      {children}
    </div>
  );
};

/**
 * Grid component for card layouts
 */
export const CardGrid = ({ children, columns = 3, gap = '20px', ...props }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: `repeat(auto-fit, minmax(300px, 1fr))`,
        gap,
      }}
      {...props}
    >
      {children}
    </div>
  );
};
