import client from './client';

/**
 * List all programs
 */
export const listPrograms = async () => {
  const response = await client.get('/programs');
  return response.data;
};

/**
 * Get program details
 */
export const getProgram = async (id) => {
  const response = await client.get(`/programs/${id}`);
  return response.data;
};

/**
 * Create new program (admin only)
 */
export const createProgram = async (programData) => {
  const response = await client.post('/programs', programData);
  return response.data;
};

/**
 * Update program (admin only)
 */
export const updateProgram = async (id, updates) => {
  const response = await client.put(`/programs/${id}`, updates);
  return response.data;
};

/**
 * Delete program (admin only)
 */
export const deleteProgram = async (id) => {
  const response = await client.delete(`/programs/${id}`);
  return response.data;
};
