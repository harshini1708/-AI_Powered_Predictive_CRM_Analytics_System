import client from './client';

/**
 * List all apprentices (paginated)
 */
export const listApprentices = async (filters = {}) => {
  const params = new URLSearchParams();
  if (filters.status) params.append('status', filters.status);
  if (filters.limit) params.append('limit', filters.limit);
  if (filters.offset) params.append('offset', filters.offset);

  const response = await client.get(`/apprentices${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get apprentice details
 */
export const getApprentice = async (id) => {
  const response = await client.get(`/apprentices/${id}`);
  return response.data;
};

/**
 * Get apprentice progress
 */
export const getApprenticeProgress = async (id) => {
  const response = await client.get(`/apprentices/${id}/progress`);
  return response.data;
};

/**
 * Create new apprentice
 */
export const createApprentice = async (apprenticeData) => {
  const response = await client.post('/apprentices', apprenticeData);
  return response.data;
};

/**
 * Update apprentice
 */
export const updateApprentice = async (id, updates) => {
  const response = await client.put(`/apprentices/${id}`, updates);
  return response.data;
};

/**
 * Deactivate apprentice
 */
export const deactivateApprentice = async (id) => {
  const response = await client.post(`/apprentices/${id}/deactivate`);
  return response.data;
};
