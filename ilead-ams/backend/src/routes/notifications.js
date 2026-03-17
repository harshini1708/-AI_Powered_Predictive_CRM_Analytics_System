import express from 'express';
import { authenticate } from '../middleware/auth.js';
import notificationService from '../services/notificationService.js';

const router = express.Router();

// All routes require authentication
router.use(authenticate);

/**
 * GET /api/notifications - Get user notifications
 */
router.get('/', async (req, res) => {
  try {
    const { limit = 20, skip = 0, unreadOnly = false, type = null } = req.query;

    const result = await notificationService.getUserNotifications(req.user.id, {
      limit: parseInt(limit),
      skip: parseInt(skip),
      unreadOnly: unreadOnly === 'true',
      type
    });

    res.json(result);
  } catch (error) {
    console.error('Error fetching notifications:', error);
    res.status(500).json({ error: 'Failed to fetch notifications' });
  }
});

/**
 * GET /api/notifications/unread - Get unread count
 */
router.get('/unread/count', async (req, res) => {
  try {
    const count = await notificationService.getUnreadCount(req.user.id);
    res.json({ unreadCount: count });
  } catch (error) {
    console.error('Error fetching unread count:', error);
    res.status(500).json({ error: 'Failed to fetch unread count' });
  }
});

/**
 * PUT /api/notifications/:id/read - Mark notification as read
 */
router.put('/:id/read', async (req, res) => {
  try {
    const notification = await notificationService.markAsRead(req.params.id, req.user.id);

    if (!notification) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    res.json(notification);
  } catch (error) {
    console.error('Error marking notification as read:', error);
    res.status(500).json({ error: 'Failed to mark as read' });
  }
});

/**
 * PUT /api/notifications/read-all - Mark all notifications as read
 */
router.put('/read-all', async (req, res) => {
  try {
    await notificationService.markAllAsRead(req.user.id);
    res.json({ message: 'All notifications marked as read' });
  } catch (error) {
    console.error('Error marking all as read:', error);
    res.status(500).json({ error: 'Failed to mark all as read' });
  }
});

/**
 * DELETE /api/notifications/:id - Delete notification
 */
router.delete('/:id', async (req, res) => {
  try {
    const deleted = await notificationService.deleteNotification(req.params.id, req.user.id);

    if (!deleted) {
      return res.status(404).json({ error: 'Notification not found' });
    }

    res.json({ message: 'Notification deleted' });
  } catch (error) {
    console.error('Error deleting notification:', error);
    res.status(500).json({ error: 'Failed to delete notification' });
  }
});

export default router;
