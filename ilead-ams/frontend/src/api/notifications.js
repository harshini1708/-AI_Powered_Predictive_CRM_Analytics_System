import client from './client';

/**
 * Get user notifications
 */
export const getNotifications = async (options = {}) => {
  const { limit = 20, skip = 0, unreadOnly = false, type = null } = options;
  const params = new URLSearchParams();
  
  if (limit) params.append('limit', limit);
  if (skip) params.append('skip', skip);
  if (unreadOnly) params.append('unreadOnly', unreadOnly);
  if (type) params.append('type', type);

  const response = await client.get(`/notifications${params ? `?${params}` : ''}`);
  return response.data;
};

/**
 * Get unread notification count
 */
export const getUnreadCount = async () => {
  const response = await client.get('/notifications/unread/count');
  return response.data;
};

/**
 * Mark notification as read
 */
export const markAsRead = async (notificationId) => {
  const response = await client.put(`/notifications/${notificationId}/read`);
  return response.data;
};

/**
 * Mark all notifications as read
 */
export const markAllAsRead = async () => {
  const response = await client.put('/notifications/read-all');
  return response.data;
};

/**
 * Delete notification
 */
export const deleteNotification = async (notificationId) => {
  const response = await client.delete(`/notifications/${notificationId}`);
  return response.data;
};
