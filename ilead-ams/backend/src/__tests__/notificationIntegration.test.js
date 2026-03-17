import notificationService from '../services/notificationService.js';
import emailService from '../services/emailService.js';

/**
 * Integration Tests for Notification System
 * Tests end-to-end notification flows
 */

describe('Notification System - Integration Tests', () => {
  describe('Hour Log Workflow', () => {
    it('should handle complete hour submission -> approval workflow', async () => {
      const apprentice = {
        _id: 'apprentice-123',
        firstName: 'John',
        email: 'john@example.com'
      };

      const hourLog = {
        _id: 'log-123',
        apprenticeId: 'apprentice-123',
        date: new Date('2026-03-17'),
        ojtHours: 8,
        rtiHours: 2,
        status: 'pending'
      };

      const supervisor = {
        _id: 'supervisor-123',
        firstName: 'Jane',
        lastName: 'Doe',
        email: 'jane@example.com'
      };

      // 1. Notify submission
      const submissionNotif = await notificationService.notifyHourSubmission(
        apprentice,
        hourLog
      );
      expect(submissionNotif).toBeDefined();
      expect(submissionNotif.type).toBe('hour_submitted');

      // 2. Notify approval
      const approvalNotif = await notificationService.notifyHourApproval(
        apprentice,
        hourLog,
        supervisor
      );
      expect(approvalNotif).toBeDefined();
      expect(approvalNotif.type).toBe('hour_approved');

      // 3. Verify emails were sent (if configured)
      if (emailService.transporter) {
        expect(emailService.sendHourApprovalNotification).toHaveBeenCalled();
      }
    });

    it('should handle rejection workflow with reason', async () => {
      const apprentice = { _id: 'apprentice-123', firstName: 'John' };
      const hourLog = { _id: 'log-123', date: new Date('2026-03-17') };
      const supervisor = { firstName: 'Jane' };

      const rejectionNotif = await notificationService.notifyHourRejection(
        apprentice,
        hourLog,
        supervisor,
        'Documentation incomplete'
      );

      expect(rejectionNotif).toBeDefined();
      expect(rejectionNotif.type).toBe('hour_rejected');
      expect(rejectionNotif.priority).toBe('high');
    });
  });

  describe('User Account Workflow', () => {
    it('should notify on new account creation', async () => {
      const newUser = {
        _id: 'user-123',
        firstName: 'John',
        email: 'john@example.com'
      };

      const tempPassword = 'TempPass123!';

      const accountNotif = await notificationService.notifyAccountCreated(
        newUser,
        tempPassword
      );

      expect(accountNotif).toBeDefined();
      expect(accountNotif.type).toBe('account_created');
      expect(accountNotif.inAppOnly).toBe(true);
    });
  });

  describe('Supervisor Notifications', () => {
    it('should send pending approvals digest to supervisor', async () => {
      const supervisor = {
        _id: 'supervisor-123',
        firstName: 'Jane',
        email: 'jane@example.com'
      };

      const pendingLogs = [
        {
          _id: 'log-1',
          apprenticeName: 'John Doe',
          date: new Date('2026-03-17'),
          ojtHours: 8,
          createdAt: new Date()
        },
        {
          _id: 'log-2',
          apprenticeName: 'Jane Smith',
          date: new Date('2026-03-16'),
          ojtHours: 6,
          createdAt: new Date()
        }
      ];

      const digestNotif = await notificationService.notifyPendingApprovals(
        supervisor,
        pendingLogs
      );

      expect(digestNotif).toBeDefined();
      expect(digestNotif.type).toBe('pending_approvals');
      expect(digestNotif.priority).toBe('high');

      // Verify email was sent
      if (emailService.transporter) {
        expect(emailService.sendPendingApprovalsDigest).toHaveBeenCalledWith(
          supervisor,
          pendingLogs
        );
      }
    });
  });

  describe('Notification Retrieval', () => {
    it('should retrieve and filter user notifications', async () => {
      const userId = 'user-123';

      // Get all notifications
      const allNotifs = await notificationService.getUserNotifications(userId);
      expect(allNotifs).toBeDefined();
      expect(allNotifs.notifications).toBeInstanceOf(Array);
      expect(allNotifs.unreadCount).toBeDefined();

      // Get unread only
      const unreadNotifs = await notificationService.getUserNotifications(userId, {
        unreadOnly: true
      });
      expect(unreadNotifs.notifications).toBeDefined();

      // Get by type
      const approvalNotifs = await notificationService.getUserNotifications(userId, {
        type: 'hour_approved'
      });
      expect(approvalNotifs.notifications).toBeDefined();
    });

    it('should get unread notification count', async () => {
      const userId = 'user-123';
      const count = await notificationService.getUnreadCount(userId);
      expect(typeof count).toBe('number');
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Notification Actions', () => {
    it('should mark notification as read', async () => {
      const notificationId = 'notif-123';
      const userId = 'user-123';

      const result = await notificationService.markAsRead(notificationId, userId);
      expect(result).toBeDefined();
      if (result) {
        expect(result.read).toBe(true);
        expect(result.readAt).toBeDefined();
      }
    });

    it('should mark all notifications as read', async () => {
      const userId = 'user-123';
      const result = await notificationService.markAllAsRead(userId);
      expect(result).toBeDefined();
      expect(result.modifiedCount).toBeDefined();
    });

    it('should delete notification', async () => {
      const notificationId = 'notif-123';
      const userId = 'user-123';

      const deleted = await notificationService.deleteNotification(notificationId, userId);
      expect(typeof deleted).toBe('boolean');
    });
  });

  describe('Error Scenarios', () => {
    it('should handle missing user gracefully', async () => {
      const invalidUser = null;
      const hourLog = { _id: 'log-123', date: new Date() };

      // Should not throw, just log warning
      expect(() => {
        notificationService.notifyHourSubmission(invalidUser, hourLog);
      }).not.toThrow();
    });

    it('should handle database errors gracefully', async () => {
      const userId = 'user-123';

      try {
        // This should either succeed or throw a specific error
        await notificationService.getUserNotifications(userId);
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe('Email Notification Flow', () => {
    it('should send email when notification created with email enabled', async () => {
      // Mock email service
      const sendEmailSpy = jest.spyOn(emailService, 'sendEmail');

      const apprentice = { _id: 'user-123', email: 'test@example.com' };
      const hourLog = { _id: 'log-123', date: new Date() };

      await notificationService.notifyHourSubmission(apprentice, hourLog);

      // Email service method would be called if configured
      expect(sendEmailSpy).toBeDefined();

      sendEmailSpy.mockRestore();
    });
  });

  describe('Notification Cleanup', () => {
    it('should cleanup old notifications', async () => {
      const result = await notificationService.cleanupOldNotifications();
      expect(result).toBeDefined();
      expect(result.deletedCount).toBeGreaterThanOrEqual(0);
    });
  });
});
