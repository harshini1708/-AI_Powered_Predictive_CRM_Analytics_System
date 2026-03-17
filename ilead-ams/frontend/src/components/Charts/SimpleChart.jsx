import React from 'react';
import { useTheme } from '../../context/ThemeContext';

/**
 * Simple Bar Chart Component - Pure SVG
 */
export const BarChart = ({ data, title, height = 300 }) => {
  const { currentTheme } = useTheme();

  if (!data || data.length === 0) return null;

  const maxValue = Math.max(...data.map(d => d.value));

  return (
    <div style={{ width: '100%', marginBottom: '20px' }}>
      {title && (
        <h3 style={{ margin: '0 0 15px 0', color: currentTheme.text }}>
          {title}
        </h3>
      )}
      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        height: `${height}px`,
        gap: '8px',
        padding: '20px 0',
        borderLeft: `2px solid ${currentTheme.border}`,
        borderBottom: `2px solid ${currentTheme.border}`
      }}>
        {data.map((item, idx) => (
          <div
            key={idx}
            style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <div
              style={{
                width: '100%',
                height: `${(item.value / maxValue) * 100}%`,
                background: item.color || currentTheme.primary,
                borderRadius: '4px 4px 0 0',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                opacity: 0.8,
              }}
              title={`${item.label}: ${item.value}`}
            />
            <small style={{
              fontSize: '11px',
              color: currentTheme.textSecondary,
              textAlign: 'center',
              wordBreak: 'break-word'
            }}>
              {item.label}
            </small>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Simple Pie Chart Component
 */
export const PieChart = ({ data, title, size = 200 }) => {
  const { currentTheme } = useTheme();

  if (!data || data.length === 0) return null;

  const total = data.reduce((sum, item) => sum + item.value, 0);
  let currentAngle = 0;

  const slices = data.map((item) => {
    const slicePercent = item.value / total;
    const sliceAngle = slicePercent * 360;
    const startAngle = currentAngle;
    const endAngle = currentAngle + sliceAngle;

    const x1 = size + size * Math.cos((startAngle * Math.PI) / 180);
    const y1 = size + size * Math.sin((startAngle * Math.PI) / 180);
    const x2 = size + size * Math.cos((endAngle * Math.PI) / 180);
    const y2 = size + size * Math.sin((endAngle * Math.PI) / 180);

    const largeArc = sliceAngle > 180 ? 1 : 0;

    const pathData = [
      `M ${size} ${size}`,
      `L ${x1} ${y1}`,
      `A ${size} ${size} 0 ${largeArc} 1 ${x2} ${y2}`,
      'Z'
    ].join(' ');

    currentAngle = endAngle;

    return {
      path: pathData,
      color: item.color || currentTheme.primary,
      label: item.label,
      value: item.value,
      percent: (slicePercent * 100).toFixed(1)
    };
  });

  return (
    <div style={{ marginBottom: '20px' }}>
      {title && (
        <h3 style={{ margin: '0 0 15px 0', color: currentTheme.text }}>
          {title}
        </h3>
      )}
      <div style={{ display: 'flex', gap: '30px', alignItems: 'center' }}>
        <svg width={size * 2} height={size * 2} style={{ maxWidth: '100%' }}>
          {slices.map((slice, idx) => (
            <path
              key={idx}
              d={slice.path}
              fill={slice.color}
              opacity="0.8"
              style={{ cursor: 'pointer' }}
            />
          ))}
        </svg>
        <div style={{ flex: 1 }}>
          {slices.map((slice, idx) => (
            <div key={idx} style={{ marginBottom: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                <div
                  style={{
                    width: '12px',
                    height: '12px',
                    background: slice.color,
                    borderRadius: '2px'
                  }}
                />
                <span style={{ fontSize: '12px', color: currentTheme.text }}>
                  {slice.label}
                </span>
              </div>
              <div style={{ fontSize: '11px', color: currentTheme.textSecondary }}>
                {slice.value} ({slice.percent}%)
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

/**
 * Progress Ring Component
 */
export const ProgressRing = ({ value, max = 100, title, size = 120 }) => {
  const { currentTheme } = useTheme();

  const radius = size / 2 - 8;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / max) * circumference;
  const percentage = Math.round((value / max) * 100);

  return (
    <div style={{ textAlign: 'center', marginBottom: '20px' }}>
      {title && (
        <h3 style={{ margin: '0 0 15px 0', color: currentTheme.text }}>
          {title}
        </h3>
      )}
      <div style={{ position: 'relative', width: size, height: size, margin: '0 auto' }}>
        <svg width={size} height={size}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={currentTheme.border}
            strokeWidth="4"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={currentTheme.success}
            strokeWidth="4"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
            style={{ transition: 'stroke-dashoffset 0.35s ease' }}
          />
        </svg>
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            textAlign: 'center'
          }}
        >
          <div style={{ fontSize: '24px', fontWeight: 'bold', color: currentTheme.text }}>
            {percentage}%
          </div>
          <div style={{ fontSize: '11px', color: currentTheme.textSecondary }}>
            {value}/{max}
          </div>
        </div>
      </div>
    </div>
  );
};
