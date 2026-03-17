import client from './client';

/**
 * Get user's reports
 */
export const getReports = async (options = {}) => {
  const { status, type, limit = 20, skip = 0 } = options;
  const params = new URLSearchParams();
  
  if (status) params.append('status', status);
  if (type) params.append('type', type);
  if (limit) params.append('limit', limit);
  if (skip) params.append('skip', skip);

  const response = await client.get(`/reports${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get single report
 */
export const getReport = async (reportId) => {
  const response = await client.get(`/reports/${reportId}`);
  return response.data;
};

/**
 * Generate report data
 */
export const generateReportData = async (reportId) => {
  const response = await client.get(`/reports/${reportId}/data`);
  return response.data;
};

/**
 * Create report
 */
export const createReport = async (reportData) => {
  const response = await client.post('/reports', reportData);
  return response.data;
};

/**
 * Update report
 */
export const updateReport = async (reportId, reportData) => {
  const response = await client.put(`/reports/${reportId}`, reportData);
  return response.data;
};

/**
 * Delete report
 */
export const deleteReport = async (reportId) => {
  const response = await client.delete(`/reports/${reportId}`);
  return response.data;
};

/**
 * Share report with user
 */
export const shareReport = async (reportId, targetUserId, permission = 'view') => {
  const response = await client.post(`/reports/${reportId}/share`, {
    targetUserId,
    permission
  });
  return response.data;
};

/**
 * Get report templates
 */
export const getReportTemplates = async () => {
  const response = await client.get('/reports/templates');
  return response.data;
};
