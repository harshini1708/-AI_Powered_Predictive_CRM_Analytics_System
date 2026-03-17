import Notification from '../models/Notification.js';
import emailService from './emailService.js';

/**
 * Notification Service - Manages in-app and email notifications
 */
class NotificationService {
  /**
   * Create and send notification
   */
  async createNotification(userId, notificationData) {
    try {
      const notification = new Notification({
        userId,
        ...notificationData
      });

      await notification.save();

      // Send email if enabled
      if (!notificationData.inAppOnly) {
        await this.sendNotificationEmail(notification, notificationData);
      }

      return notification;
    } catch (error) {
      console.error('Error creating notification:', error);
      throw error;
    }
  }

  /**
   * Send notification via email
   */
  async sendNotificationEmail(notification, data) {
    // This would be implemented based on notification type
    // For now, just log that it would be sent
    console.log(`📧 Email notification queued: ${notification.type} to ${notification.userId}`);
  }

  /**
   * Get user notifications
   */
  async getUserNotifications(userId, options = {}) {
    const {
      limit = 20,
      skip = 0,
      unreadOnly = false,
      type = null
    } = options;

    const query = { userId };

    if (unreadOnly) {
      query.read = false;
    }

    if (type) {
      query.type = type;
    }

    const notifications = await Notification
      .find(query)
      .sort({ createdAt: -1 })
      .limit(limit)
      .skip(skip)
      .lean();

    const total = await Notification.countDocuments(query);

    return {
      notifications,
      total,
      unreadCount: await Notification.countDocuments({ ...query, read: false })
    };
  }

  /**
   * Mark notification as read
   */
  async markAsRead(notificationId, userId) {
    const notification = await Notification.findOneAndUpdate(
      { _id: notificationId, userId },
      {
        read: true,
        readAt: new Date()
      },
      { new: true }
    );

    return notification;
  }

  /**
   * Mark all notifications as read for user
   */
  async markAllAsRead(userId) {
    const result = await Notification.updateMany(
      { userId, read: false },
      {
        read: true,
        readAt: new Date()
      }
    );

    return result;
  }

  /**
   * Delete notification
   */
  async deleteNotification(notificationId, userId) {
    const result = await Notification.deleteOne({
      _id: notificationId,
      userId
    });

    return result.deletedCount > 0;
  }

  /**
   * Get unread count for user
   */
  async getUnreadCount(userId) {
    const count = await Notification.countDocuments({
      userId,
      read: false
    });

    return count;
  }

  /**
   * Notify hour submission
   */
  async notifyHourSubmission(apprentice, hourLog) {
    return this.createNotification(apprentice._id, {
      type: 'hour_submitted',
      title: 'Hour Log Submitted',
      message: `Your hour log for ${new Date(hourLog.date).toLocaleDateString()} has been submitted for approval.`,
      relatedId: hourLog._id,
      relatedType: 'HourLog',
      priority: 'normal',
      actionUrl: `/hours/${hourLog._id}`,
      actionLabel: 'View Details'
    });
  }

  /**
   * Notify hour approval
   */
  async notifyHourApproval(apprentice, hourLog, supervisor) {
    await this.createNotification(apprentice._id, {
      type: 'hour_approved',
      title: 'Hour Log Approved ✅',
      message: `Your hour log for ${new Date(hourLog.date).toLocaleDateString()} has been approved by ${supervisor.firstName}.`,
      relatedId: hourLog._id,
      relatedType: 'HourLog',
      priority: 'normal',
      actionUrl: `/hours/${hourLog._id}`,
      actionLabel: 'View Details'
    });

    // Also send email notification
    await emailService.sendHourApprovalNotification(apprentice, hourLog, supervisor);
  }

  /**
   * Notify hour rejection
   */
  async notifyHourRejection(apprentice, hourLog, supervisor, reason) {
    await this.createNotification(apprentice._id, {
      type: 'hour_rejected',
      title: 'Hour Log Requires Revision ⚠️',
      message: `Your hour log for ${new Date(hourLog.date).toLocaleDateString()} requires revision. Reason: ${reason}`,
      relatedId: hourLog._id,
      relatedType: 'HourLog',
      priority: 'high',
      actionUrl: `/hours/${hourLog._id}`,
      actionLabel: 'Review & Resubmit'
    });

    // Also send email notification
    await emailService.sendHourRejectionNotification(apprentice, hourLog, supervisor, reason);
  }

  /**
   * Notify pending approvals (batch)
   */
  async notifyPendingApprovals(supervisor, pendingLogs) {
    await this.createNotification(supervisor._id, {
      type: 'pending_approvals',
      title: `${pendingLogs.length} Pending Approvals`,
      message: `You have ${pendingLogs.length} hour log(s) pending approval.`,
      priority: 'high',
      actionUrl: '/approvals',
      actionLabel: 'Review Approvals',
      inAppOnly: false
    });

    // Also send email
    await emailService.sendPendingApprovalsDigest(supervisor, pendingLogs);
  }

  /**
   * Notify account created
   */
  async notifyAccountCreated(user, tempPassword) {
    await emailService.sendAccountCreatedNotification(user, tempPassword);

    // Also create in-app notification
    return this.createNotification(user._id, {
      type: 'account_created',
      title: 'Welcome to I-LEAD AMS',
      message: 'Your account has been created. Please log in with your credentials.',
      priority: 'normal',
      actionUrl: '/login',
      actionLabel: 'Log In',
      inAppOnly: true
    });
  }

  /**
   * Notify admin alert
   */
  async notifyAdminAlert(adminId, alertType, details) {
    return this.createNotification(adminId, {
      type: 'admin_alert',
      title: `System Alert: ${alertType}`,
      message: details.message || `An alert has been triggered: ${alertType}`,
      priority: 'urgent',
      metadata: details,
      inAppOnly: false
    });
  }

  /**
   * Clean up old notifications (older than 30 days)
   */
  async cleanupOldNotifications() {
    const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);

    const result = await Notification.deleteMany({
      createdAt: { $lt: thirtyDaysAgo }
    });

    console.log(`Cleaned up ${result.deletedCount} old notifications`);
    return result;
  }
}

export default new NotificationService();
