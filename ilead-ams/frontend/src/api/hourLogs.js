import client from './client';

/**
 * Submit hour log
 */
export const submitHours = async (hourLogData) => {
  const response = await client.post('/hour-logs/submit', hourLogData);
  return response.data;
};

/**
 * List hour logs (filtered by role)
 */
export const listHourLogs = async (filters = {}) => {
  const params = new URLSearchParams();
  if (filters.status) params.append('status', filters.status);
  if (filters.apprenticeId) params.append('apprenticeId', filters.apprenticeId);
  if (filters.dateFrom) params.append('dateFrom', filters.dateFrom);
  if (filters.dateTo) params.append('dateTo', filters.dateTo);
  if (filters.domain) params.append('domain', filters.domain);
  if (filters.minHours) params.append('minHours', filters.minHours);
  if (filters.maxHours) params.append('maxHours', filters.maxHours);
  if (filters.sortBy) params.append('sortBy', filters.sortBy);
  if (filters.limit) params.append('limit', filters.limit);
  if (filters.offset) params.append('offset', filters.offset);

  const response = await client.get(`/hour-logs${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get hour log details
 */
export const getHourLog = async (id) => {
  const response = await client.get(`/hour-logs/${id}`);
  return response.data;
};

/**
 * Approve hour log with rubric scoring
 */
export const approveHourLog = async (id, approvalData) => {
  const response = await client.post(`/hour-logs/${id}/approve`, approvalData);
  return response.data;
};

/**
 * Reject hour log
 */
export const rejectHourLog = async (id, rejectionData) => {
  const response = await client.post(`/hour-logs/${id}/reject`, rejectionData);
  return response.data;
};

/**
 * Get hour logs statistics for apprentice
 */
export const getApprenticeHourStats = async (apprenticeId) => {
  const response = await client.get(`/hour-logs/apprentice/${apprenticeId}/stats`);
  return response.data;
};
