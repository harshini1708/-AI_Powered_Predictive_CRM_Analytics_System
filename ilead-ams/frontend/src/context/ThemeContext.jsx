import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

/**
 * Light theme colors
 */
const lightTheme = {
  name: 'light',
  primary: '#003d82',
  primaryDark: '#002855',
  primaryLight: '#1a5ca5',
  secondary: '#e84c1f',
  success: '#28a745',
  warning: '#ffc107',
  danger: '#dc3545',
  info: '#17a2b8',
  
  // Neutral colors
  white: '#ffffff',
  black: '#000000',
  gray50: '#f8f9fa',
  gray100: '#f1f3f5',
  gray200: '#e9ecef',
  gray300: '#dee2e6',
  gray400: '#ced4da',
  gray500: '#adb5bd',
  gray600: '#6c757d',
  gray700: '#495057',
  gray800: '#343a40',
  gray900: '#212529',
  
  // Backgrounds
  background: '#ffffff',
  backgroundSecondary: '#f8f9fa',
  surfaceLight: '#fafbfc',
  
  // Text
  text: '#212529',
  textSecondary: '#6c757d',
  textLight: '#adb5bd',
  
  // Borders
  border: '#e9ecef',
  borderLight: '#f1f3f5',
  
  // Shadows
  shadowXs: '0 1px 2px rgba(0,0,0,0.05)',
  shadowSm: '0 1px 3px rgba(0,0,0,0.1)',
  shadowMd: '0 4px 6px rgba(0,0,0,0.1)',
  shadowLg: '0 10px 15px rgba(0,0,0,0.1)',
  shadowXl: '0 20px 25px rgba(0,0,0,0.1)',
};

/**
 * Dark theme colors
 */
const darkTheme = {
  name: 'dark',
  primary: '#4a9eff',
  primaryDark: '#2979be',
  primaryLight: '#7bb4ff',
  secondary: '#ff8844',
  success: '#4ade80',
  warning: '#facc15',
  danger: '#ff6b6b',
  info: '#22d3ee',
  
  // Neutral colors
  white: '#1a1a1a',
  black: '#ffffff',
  gray50: '#0f0f0f',
  gray100: '#1a1a1a',
  gray200: '#2d2d2d',
  gray300: '#404040',
  gray400: '#525252',
  gray500: '#737373',
  gray600: '#a3a3a3',
  gray700: '#d4d4d8',
  gray800: '#e4e4e7',
  gray900: '#fafafa',
  
  // Backgrounds
  background: '#0f172a',
  backgroundSecondary: '#1a1f35',
  surfaceLight: '#1e2847',
  
  // Text
  text: '#f1f5f9',
  textSecondary: '#cbd5e1',
  textLight: '#94a3b8',
  
  // Borders
  border: '#334155',
  borderLight: '#1e293b',
  
  // Shadows
  shadowXs: '0 1px 2px rgba(0,0,0,0.3)',
  shadowSm: '0 1px 3px rgba(0,0,0,0.4)',
  shadowMd: '0 4px 6px rgba(0,0,0,0.5)',
  shadowLg: '0 10px 15px rgba(0,0,0,0.6)',
  shadowXl: '0 20px 25px rgba(0,0,0,0.7)',
};

/**
 * Theme Provider Component
 */
export const ThemeProvider = ({ children }) => {
  const [isDarkMode, setIsDarkMode] = useState(() => {
    // Check localStorage or system preference
    const saved = localStorage.getItem('theme-mode');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const currentTheme = isDarkMode ? darkTheme : lightTheme;

  useEffect(() => {
    localStorage.setItem('theme-mode', isDarkMode ? 'dark' : 'light');
    // Update document class for CSS-based theming
    document.documentElement.setAttribute('data-theme', isDarkMode ? 'dark' : 'light');
  }, [isDarkMode]);

  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  return (
    <ThemeContext.Provider value={{ currentTheme, isDarkMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

/**
 * Hook to use theme
 */
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
};
