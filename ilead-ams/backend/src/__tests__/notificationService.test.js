import notificationService from '../services/notificationService.js';
import Notification from '../models/Notification.js';
import emailService from '../services/emailService.js';

// Mock dependencies
jest.mock('../models/Notification.js');
jest.mock('../services/emailService.js');

describe('Notification Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Create Notification', () => {
    it('should create a notification', async () => {
      const mockNotification = {
        _id: 'notification-id',
        userId: 'user-id',
        type: 'hour_submitted',
        save: jest.fn().mockResolvedValue(true)
      };

      Notification.mockImplementation(() => mockNotification);

      const result = await notificationService.createNotification('user-id', {
        type: 'hour_submitted',
        title: 'Hour Submitted',
        message: 'Your hour log has been submitted'
      });

      expect(mockNotification.save).toHaveBeenCalled();
      expect(result).toBeDefined();
    });

    it('should not send email if inAppOnly is true', async () => {
      const mockNotification = {
        save: jest.fn().mockResolvedValue(true)
      };

      Notification.mockImplementation(() => mockNotification);

      await notificationService.createNotification('user-id', {
        type: 'hour_submitted',
        title: 'Title',
        message: 'Message',
        inAppOnly: true
      });

      expect(emailService.sendEmail).not.toHaveBeenCalled();
    });

    it('should handle creation errors', async () => {
      Notification.mockImplementation(() => {
        throw new Error('Database error');
      });

      await expect(
        notificationService.createNotification('user-id', {
          type: 'hour_submitted',
          title: 'Title',
          message: 'Message'
        })
      ).rejects.toThrow('Database error');
    });
  });

  describe('Get User Notifications', () => {
    it('should retrieve user notifications with pagination', async () => {
      const mockNotifications = [
        { _id: '1', userId: 'user-id', type: 'hour_submitted', read: false },
        { _id: '2', userId: 'user-id', type: 'hour_approved', read: true }
      ];

      Notification.find = jest.fn().mockReturnValue({
        sort: jest.fn().mockReturnValue({
          limit: jest.fn().mockReturnValue({
            skip: jest.fn().mockReturnValue({
              lean: jest.fn().mockResolvedValue(mockNotifications)
            })
          })
        })
      });

      Notification.countDocuments = jest
        .fn()
        .mockResolvedValueOnce(2)
        .mockResolvedValueOnce(1);

      const result = await notificationService.getUserNotifications('user-id', {
        limit: 20,
        skip: 0
      });

      expect(result.notifications).toHaveLength(2);
      expect(result.total).toBe(2);
      expect(result.unreadCount).toBe(1);
    });

    it('should filter by unread only', async () => {
      Notification.find = jest.fn().mockReturnValue({
        sort: jest.fn().mockReturnValue({
          limit: jest.fn().mockReturnValue({
            skip: jest.fn().mockReturnValue({
              lean: jest.fn().mockResolvedValue([])
            })
          })
        })
      });

      Notification.countDocuments = jest.fn().mockResolvedValue(0);

      const result = await notificationService.getUserNotifications('user-id', {
        unreadOnly: true
      });

      expect(Notification.find).toHaveBeenCalledWith(
        expect.objectContaining({
          userId: 'user-id',
          read: false
        })
      );
    });

    it('should filter by notification type', async () => {
      Notification.find = jest.fn().mockReturnValue({
        sort: jest.fn().mockReturnValue({
          limit: jest.fn().mockReturnValue({
            skip: jest.fn().mockReturnValue({
              lean: jest.fn().mockResolvedValue([])
            })
          })
        })
      });

      Notification.countDocuments = jest.fn().mockResolvedValue(0);

      await notificationService.getUserNotifications('user-id', {
        type: 'hour_approved'
      });

      expect(Notification.find).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'hour_approved'
        })
      );
    });
  });

  describe('Mark as Read', () => {
    it('should mark notification as read', async () => {
      const mockResult = {
        _id: 'notification-id',
        read: true,
        readAt: expect.any(Date)
      };

      Notification.findOneAndUpdate = jest.fn().mockResolvedValue(mockResult);

      const result = await notificationService.markAsRead('notification-id', 'user-id');

      expect(Notification.findOneAndUpdate).toHaveBeenCalledWith(
        { _id: 'notification-id', userId: 'user-id' },
        expect.objectContaining({
          read: true,
          readAt: expect.any(Date)
        }),
        { new: true }
      );

      expect(result.read).toBe(true);
    });

    it('should return null if notification not found', async () => {
      Notification.findOneAndUpdate = jest.fn().mockResolvedValue(null);

      const result = await notificationService.markAsRead('invalid-id', 'user-id');

      expect(result).toBeNull();
    });
  });

  describe('Mark All as Read', () => {
    it('should mark all user notifications as read', async () => {
      Notification.updateMany = jest.fn().mockResolvedValue({
        modifiedCount: 5
      });

      const result = await notificationService.markAllAsRead('user-id');

      expect(Notification.updateMany).toHaveBeenCalledWith(
        { userId: 'user-id', read: false },
        expect.objectContaining({
          read: true,
          readAt: expect.any(Date)
        })
      );

      expect(result.modifiedCount).toBe(5);
    });
  });

  describe('Delete Notification', () => {
    it('should delete a notification', async () => {
      Notification.deleteOne = jest.fn().mockResolvedValue({ deletedCount: 1 });

      const result = await notificationService.deleteNotification('notification-id', 'user-id');

      expect(Notification.deleteOne).toHaveBeenCalledWith({
        _id: 'notification-id',
        userId: 'user-id'
      });

      expect(result).toBe(true);
    });

    it('should return false if notification not found', async () => {
      Notification.deleteOne = jest.fn().mockResolvedValue({ deletedCount: 0 });

      const result = await notificationService.deleteNotification('invalid-id', 'user-id');

      expect(result).toBe(false);
    });
  });

  describe('Get Unread Count', () => {
    it('should return unread notification count', async () => {
      Notification.countDocuments = jest.fn().mockResolvedValue(3);

      const count = await notificationService.getUnreadCount('user-id');

      expect(Notification.countDocuments).toHaveBeenCalledWith({
        userId: 'user-id',
        read: false
      });

      expect(count).toBe(3);
    });

    it('should return 0 if no unread notifications', async () => {
      Notification.countDocuments = jest.fn().mockResolvedValue(0);

      const count = await notificationService.getUnreadCount('user-id');

      expect(count).toBe(0);
    });
  });

  describe('Notify Hour Submission', () => {
    it('should create notification for hour submission', async () => {
      const mockNotification = {
        save: jest.fn().mockResolvedValue(true)
      };

      Notification.mockImplementation(() => mockNotification);

      const apprentice = { _id: 'apprentice-id' };
      const hourLog = { _id: 'log-id', date: new Date('2026-03-17') };

      await notificationService.notifyHourSubmission(apprentice, hourLog);

      expect(mockNotification.save).toHaveBeenCalled();
    });
  });

  describe('Notify Hour Approval', () => {
    it('should create notification and send email for approval', async () => {
      const mockNotification = {
        save: jest.fn().mockResolvedValue(true)
      };

      Notification.mockImplementation(() => mockNotification);

      const apprentice = { _id: 'apprentice-id' };
      const hourLog = { _id: 'log-id', date: new Date('2026-03-17') };
      const supervisor = { firstName: 'Jane' };

      await notificationService.notifyHourApproval(apprentice, hourLog, supervisor);

      expect(emailService.sendHourApprovalNotification).toHaveBeenCalledWith(
        apprentice,
        hourLog,
        supervisor
      );
    });
  });

  describe('Cleanup Old Notifications', () => {
    it('should delete notifications older than 30 days', async () => {
      Notification.deleteMany = jest.fn().mockResolvedValue({ deletedCount: 10 });

      const result = await notificationService.cleanupOldNotifications();

      expect(Notification.deleteMany).toHaveBeenCalledWith(
        expect.objectContaining({
          createdAt: expect.any(Object)
        })
      );

      expect(result.deletedCount).toBe(10);
    });
  });
});
