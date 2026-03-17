import request from 'supertest';
import express from 'express';
import notificationsRouter from '../routes/notifications.js';
import notificationService from '../services/notificationService.js';

// Mock dependencies
jest.mock('../services/notificationService.js');
jest.mock('../middleware/auth.js', () => ({
  authenticate: (req, res, next) => {
    req.user = { id: 'test-user-id' };
    next();
  }
}));

describe('Notification Routes', () => {
  let app;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use('/api/notifications', notificationsRouter);
    jest.clearAllMocks();
  });

  describe('GET /api/notifications', () => {
    it('should retrieve notifications for user', async () => {
      const mockNotifications = [
        { _id: '1', title: 'Test', read: false },
        { _id: '2', title: 'Test 2', read: true }
      ];

      notificationService.getUserNotifications.mockResolvedValue({
        notifications: mockNotifications,
        total: 2,
        unreadCount: 1
      });

      const response = await request(app)
        .get('/api/notifications')
        .expect(200);

      expect(response.body.notifications).toHaveLength(2);
      expect(response.body.total).toBe(2);
      expect(response.body.unreadCount).toBe(1);
    });

    it('should support pagination', async () => {
      notificationService.getUserNotifications.mockResolvedValue({
        notifications: [],
        total: 100,
        unreadCount: 5
      });

      const response = await request(app)
        .get('/api/notifications?limit=10&skip=20')
        .expect(200);

      expect(notificationService.getUserNotifications).toHaveBeenCalledWith(
        'test-user-id',
        expect.objectContaining({
          limit: 10,
          skip: 20
        })
      );
    });

    it('should filter by unread only', async () => {
      notificationService.getUserNotifications.mockResolvedValue({
        notifications: [],
        total: 5,
        unreadCount: 5
      });

      await request(app)
        .get('/api/notifications?unreadOnly=true')
        .expect(200);

      expect(notificationService.getUserNotifications).toHaveBeenCalledWith(
        'test-user-id',
        expect.objectContaining({
          unreadOnly: true
        })
      );
    });
  });

  describe('GET /api/notifications/unread/count', () => {
    it('should return unread notification count', async () => {
      notificationService.getUnreadCount.mockResolvedValue(5);

      const response = await request(app)
        .get('/api/notifications/unread/count')
        .expect(200);

      expect(response.body.unreadCount).toBe(5);
    });
  });

  describe('PUT /api/notifications/:id/read', () => {
    it('should mark notification as read', async () => {
      const mockNotification = { _id: '1', read: true };

      notificationService.markAsRead.mockResolvedValue(mockNotification);

      const response = await request(app)
        .put('/api/notifications/1/read')
        .expect(200);

      expect(response.body.read).toBe(true);
      expect(notificationService.markAsRead).toHaveBeenCalledWith('1', 'test-user-id');
    });

    it('should return 404 if notification not found', async () => {
      notificationService.markAsRead.mockResolvedValue(null);

      await request(app)
        .put('/api/notifications/invalid-id/read')
        .expect(404);
    });
  });

  describe('PUT /api/notifications/read-all', () => {
    it('should mark all notifications as read', async () => {
      notificationService.markAllAsRead.mockResolvedValue({});

      const response = await request(app)
        .put('/api/notifications/read-all')
        .expect(200);

      expect(response.body.message).toContain('marked as read');
      expect(notificationService.markAllAsRead).toHaveBeenCalledWith('test-user-id');
    });
  });

  describe('DELETE /api/notifications/:id', () => {
    it('should delete notification', async () => {
      notificationService.deleteNotification.mockResolvedValue(true);

      const response = await request(app)
        .delete('/api/notifications/1')
        .expect(200);

      expect(response.body.message).toContain('deleted');
      expect(notificationService.deleteNotification).toHaveBeenCalledWith('1', 'test-user-id');
    });

    it('should return 404 if notification not found', async () => {
      notificationService.deleteNotification.mockResolvedValue(false);

      await request(app)
        .delete('/api/notifications/invalid-id')
        .expect(404);
    });
  });

  describe('Error Handling', () => {
    it('should handle service errors', async () => {
      notificationService.getUserNotifications.mockRejectedValue(new Error('Database error'));

      const response = await request(app)
        .get('/api/notifications')
        .expect(500);

      expect(response.body.error).toBeDefined();
    });
  });
});
