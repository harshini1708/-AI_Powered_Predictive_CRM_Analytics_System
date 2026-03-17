import client from './client';

/**
 * List all users (admin only)
 */
export const listUsers = async (filters = {}) => {
  const params = new URLSearchParams();
  if (filters.role) params.append('role', filters.role);
  if (filters.status) params.append('status', filters.status);
  if (filters.limit) params.append('limit', filters.limit);
  if (filters.offset) params.append('offset', filters.offset);

  const response = await client.get(`/users${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get user details
 */
export const getUser = async (id) => {
  const response = await client.get(`/users/${id}`);
  return response.data;
};

/**
 * Update user
 */
export const updateUser = async (id, updates) => {
  const response = await client.put(`/users/${id}`, updates);
  return response.data;
};

/**
 * Delete user (admin only)
 */
export const deleteUser = async (id) => {
  const response = await client.delete(`/users/${id}`);
  return response.data;
};

/**
 * Approve pending user
 */
export const approveUser = async (id, pinVerification = false) => {
  const response = await client.post(`/users/${id}/approve`, { pinVerification });
  return response.data;
};

/**
 * Deactivate user
 */
export const deactivateUser = async (id) => {
  const response = await client.post(`/users/${id}/deactivate`);
  return response.data;
};

/**
 * Create new user (super admin only)
 */
export const createUser = async (userData) => {
  const response = await client.post('/users', userData);
  return response.data;
};
