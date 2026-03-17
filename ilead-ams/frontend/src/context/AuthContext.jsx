import React, { createContext, useState, useCallback, useEffect } from 'react';
import * as authApi from '../api/auth';
import { setTokens, clearTokens, getAccessToken, getRefreshToken } from '../api/client';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Initialize auth from stored tokens
  useEffect(() => {
    const initializeAuth = async () => {
      const token = getAccessToken();
      if (token) {
        try {
          const response = await authApi.getCurrentUser();
          setUser(response.user);
          setIsAuthenticated(true);
        } catch (err) {
          clearTokens();
          setUser(null);
          setIsAuthenticated(false);
        }
      }
      setLoading(false);
    };

    initializeAuth();
  }, []);

  const register = useCallback(async (userData) => {
    try {
      setLoading(true);
      setError(null);
      const response = await authApi.register(userData);
      setTokens(response.accessToken, response.refreshToken);
      setUser(response.user);
      setIsAuthenticated(true);
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Registration failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const login = useCallback(async (email, password) => {
    try {
      setLoading(true);
      setError(null);
      const response = await authApi.login(email, password);
      setTokens(response.accessToken, response.refreshToken);
      setUser(response.user);
      setIsAuthenticated(true);
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Login failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const verifyPin = useCallback(async (pin) => {
    try {
      setLoading(true);
      setError(null);
      const response = await authApi.verifyPin(pin);
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'PIN verification failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      clearTokens();
      setUser(null);
      setIsAuthenticated(false);
      setError(null);
    }
  }, []);

  const forgotPassword = useCallback(async (email) => {
    try {
      setLoading(true);
      setError(null);
      const response = await authApi.forgotPassword(email);
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Password reset request failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const resetPassword = useCallback(async (token, newPassword) => {
    try {
      setLoading(true);
      setError(null);
      const response = await authApi.resetPassword(token, newPassword);
      return response;
    } catch (err) {
      setError(err.response?.data?.message || 'Password reset failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const value = {
    user,
    loading,
    error,
    isAuthenticated,
    register,
    login,
    verifyPin,
    logout,
    forgotPassword,
    resetPassword,
    clearError,
    setUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
