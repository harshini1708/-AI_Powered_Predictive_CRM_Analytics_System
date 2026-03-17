import React from 'react';
import { useTheme } from '../../context/ThemeContext';

/**
 * Input Component - Text, email, password, etc.
 */
export const Input = ({
  label,
  type = 'text',
  placeholder,
  value,
  onChange,
  error,
  disabled = false,
  required = false,
  helperText,
  size = 'md',
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const inputStyle = {
    width: '100%',
    padding: size === 'sm' ? '8px 12px' : size === 'lg' ? '12px 16px' : '10px 14px',
    fontSize: size === 'sm' ? '12px' : size === 'lg' ? '16px' : '14px',
    border: `1px solid ${error ? currentTheme.danger : currentTheme.border}`,
    borderRadius: '6px',
    background: currentTheme.background,
    color: currentTheme.text,
    transition: 'all 0.2s ease',
    boxSizing: 'border-box',
    ...style
  };

  return (
    <div style={{ marginBottom: '12px' }}>
      {label && (
        <label style={{
          display: 'block',
          marginBottom: '6px',
          color: currentTheme.text,
          fontSize: '14px',
          fontWeight: '500'
        }}>
          {label}
          {required && <span style={{ color: currentTheme.danger }}> *</span>}
        </label>
      )}
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        disabled={disabled}
        style={inputStyle}
        {...props}
      />
      {error && (
        <p style={{
          margin: '6px 0 0 0',
          color: currentTheme.danger,
          fontSize: '12px'
        }}>
          {error}
        </p>
      )}
      {helperText && !error && (
        <p style={{
          margin: '6px 0 0 0',
          color: currentTheme.textSecondary,
          fontSize: '12px'
        }}>
          {helperText}
        </p>
      )}
    </div>
  );
};

/**
 * Textarea Component
 */
export const Textarea = ({
  label,
  placeholder,
  value,
  onChange,
  error,
  disabled = false,
  required = false,
  rows = 4,
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const textareaStyle = {
    width: '100%',
    padding: '10px 14px',
    fontSize: '14px',
    border: `1px solid ${error ? currentTheme.danger : currentTheme.border}`,
    borderRadius: '6px',
    background: currentTheme.background,
    color: currentTheme.text,
    fontFamily: 'inherit',
    resize: 'vertical',
    minHeight: `${rows * 24}px`,
    boxSizing: 'border-box',
    ...style
  };

  return (
    <div style={{ marginBottom: '12px' }}>
      {label && (
        <label style={{
          display: 'block',
          marginBottom: '6px',
          color: currentTheme.text,
          fontSize: '14px',
          fontWeight: '500'
        }}>
          {label}
          {required && <span style={{ color: currentTheme.danger }}> *</span>}
        </label>
      )}
      <textarea
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        disabled={disabled}
        rows={rows}
        style={textareaStyle}
        {...props}
      />
      {error && (
        <p style={{
          margin: '6px 0 0 0',
          color: currentTheme.danger,
          fontSize: '12px'
        }}>
          {error}
        </p>
      )}
    </div>
  );
};

/**
 * Select Component
 */
export const Select = ({
  label,
  value,
  onChange,
  options = [],
  error,
  disabled = false,
  required = false,
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  const selectStyle = {
    width: '100%',
    padding: '10px 14px',
    fontSize: '14px',
    border: `1px solid ${error ? currentTheme.danger : currentTheme.border}`,
    borderRadius: '6px',
    background: currentTheme.background,
    color: currentTheme.text,
    cursor: disabled ? 'not-allowed' : 'pointer',
    ...style
  };

  return (
    <div style={{ marginBottom: '12px' }}>
      {label && (
        <label style={{
          display: 'block',
          marginBottom: '6px',
          color: currentTheme.text,
          fontSize: '14px',
          fontWeight: '500'
        }}>
          {label}
          {required && <span style={{ color: currentTheme.danger }}> *</span>}
        </label>
      )}
      <select
        value={value}
        onChange={onChange}
        disabled={disabled}
        style={selectStyle}
        {...props}
      >
        <option value="">Select an option</option>
        {options.map((opt, idx) => (
          <option key={idx} value={opt.value || opt}>
            {opt.label || opt}
          </option>
        ))}
      </select>
      {error && (
        <p style={{
          margin: '6px 0 0 0',
          color: currentTheme.danger,
          fontSize: '12px'
        }}>
          {error}
        </p>
      )}
    </div>
  );
};

/**
 * Checkbox Component
 */
export const Checkbox = ({
  label,
  checked = false,
  onChange,
  disabled = false,
  style = {},
  ...props
}) => {
  const { currentTheme } = useTheme();

  return (
    <label style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      marginBottom: '12px',
      ...style
    }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
        {...props}
      />
      <span style={{ color: currentTheme.text, fontSize: '14px' }}>
        {label}
      </span>
    </label>
  );
};
