import client from './client';

/**
 * Get system health status
 */
export const getSystemHealth = async () => {
  const response = await client.get('/admin/health');
  return response.data;
};

/**
 * Get user statistics
 */
export const getUserStatistics = async () => {
  const response = await client.get('/admin/statistics/users');
  return response.data;
};

/**
 * Get system statistics
 */
export const getSystemStatistics = async () => {
  const response = await client.get('/admin/statistics/system');
  return response.data;
};

/**
 * Get audit logs
 */
export const getAuditLogs = async (options = {}) => {
  const { limit = 50, skip = 0, userId = null } = options;
  const params = new URLSearchParams();
  
  if (limit) params.append('limit', limit);
  if (skip) params.append('skip', skip);
  if (userId) params.append('userId', userId);

  const response = await client.get(`/admin/audit-logs${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get system settings
 */
export const getSystemSettings = async () => {
  const response = await client.get('/admin/settings');
  return response.data;
};

/**
 * Update system settings
 */
export const updateSystemSettings = async (settings) => {
  const response = await client.put('/admin/settings', settings);
  return response.data;
};

/**
 * Get database status
 */
export const getDatabaseStatus = async () => {
  const response = await client.get('/admin/database/status');
  return response.data;
};

/**
 * Get backup status
 */
export const getBackupStatus = async () => {
  const response = await client.get('/admin/backup/status');
  return response.data;
};

/**
 * Trigger database backup
 */
export const triggerBackup = async () => {
  const response = await client.post('/admin/backup/trigger');
  return response.data;
};

/**
 * Get system logs
 */
export const getSystemLogs = async (options = {}) => {
  const { limit = 100, level = null } = options;
  const params = new URLSearchParams();
  
  if (limit) params.append('limit', limit);
  if (level) params.append('level', level);

  const response = await client.get(`/admin/logs${params ? `?${params}` : ''}`);
  return response.data;
};
