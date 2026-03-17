import client from './client';

/**
 * Get dashboard statistics
 */
export const getDashboardStats = async () => {
  const response = await client.get('/analytics/dashboard');
  return response.data;
};

/**
 * Get OJT domain progress breakdown
 */
export const getDomainProgress = async () => {
  const response = await client.get('/analytics/domain-progress');
  return response.data;
};

/**
 * Get competency heat map (rubric scores by domain)
 */
export const getCompetencyHeatMap = async () => {
  const response = await client.get('/analytics/competency-heat-map');
  return response.data;
};

/**
 * Get detailed progress for specific apprentice
 */
export const getApprenticeAnalytics = async (apprenticeId) => {
  const response = await client.get(`/analytics/apprentices/${apprenticeId}`);
  return response.data;
};

/**
 * Get progress report (PDF ready format)
 */
export const getProgressReport = async (apprenticeId) => {
  const response = await client.get(`/analytics/reports/progress/${apprenticeId}`);
  return response.data;
};

/**
 * Get supervision ratio report
 */
export const getSupervisionReport = async () => {
  const response = await client.get('/analytics/reports/supervision');
  return response.data;
};
