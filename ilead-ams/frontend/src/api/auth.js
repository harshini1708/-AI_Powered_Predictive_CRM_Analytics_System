import client from './client';

/**
 * Register a new user
 */
export const register = async (userData) => {
  const response = await client.post('/auth/register', userData);
  return response.data;
};

/**
 * Login user with email and password
 */
export const login = async (email, password) => {
  const response = await client.post('/auth/login', { email, password });
  return response.data;
};

/**
 * Verify super admin PIN
 */
export const verifyPin = async (pin) => {
  const response = await client.post('/auth/verify-pin', { pin });
  return response.data;
};

/**
 * Refresh access token
 */
export const refreshToken = async (refreshToken) => {
  const response = await client.post('/auth/refresh', { refreshToken });
  return response.data;
};

/**
 * Get current logged-in user
 */
export const getCurrentUser = async () => {
  const response = await client.get('/auth/me');
  return response.data;
};

/**
 * Logout (clear tokens on frontend)
 */
export const logout = async () => {
  try {
    await client.post('/auth/logout');
  } catch (error) {
    console.error('Logout error:', error);
  }
};

/**
 * Request password reset
 */
export const forgotPassword = async (email) => {
  const response = await client.post('/auth/forgot-password', { email });
  return response.data;
};

/**
 * Reset password with token
 */
export const resetPassword = async (token, newPassword) => {
  const response = await client.post('/auth/reset-password', { token, newPassword });
  return response.data;
};
