import mongoose from 'mongoose';

const notificationSchema = new mongoose.Schema(
  {
    // Recipient
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User',
      required: true,
      index: true
    },

    // Notification details
    type: {
      type: String,
      enum: [
        'hour_submitted',
        'hour_approved',
        'hour_rejected',
        'account_created',
        'password_reset',
        'admin_alert',
        'weekly_summary',
        'pending_approvals',
        'system_notification'
      ],
      required: true
    },

    title: {
      type: String,
      required: true
    },

    message: {
      type: String,
      required: true
    },

    // Related data
    relatedId: {
      type: mongoose.Schema.Types.ObjectId,
      description: 'ID of related resource (HourLog, User, etc.)'
    },

    relatedType: {
      type: String,
      enum: ['HourLog', 'User', 'Apprentice', 'System'],
      default: 'System'
    },

    // Status
    read: {
      type: Boolean,
      default: false,
      index: true
    },

    readAt: Date,

    // Delivery status
    emailSent: {
      type: Boolean,
      default: false
    },

    emailSentAt: Date,

    inAppOnly: {
      type: Boolean,
      default: false,
      description: 'If true, no email will be sent'
    },

    // Action link
    actionUrl: String,

    actionLabel: String,

    // Priority
    priority: {
      type: String,
      enum: ['low', 'normal', 'high', 'urgent'],
      default: 'normal'
    },

    // Metadata
    metadata: mongoose.Schema.Types.Mixed,

    // TTL for auto-deletion (30 days default)
    expiresAt: {
      type: Date,
      default: () => new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      index: { expires: 0 }
    }
  },
  {
    timestamps: true
  }
);

// Indexes
notificationSchema.index({ userId: 1, createdAt: -1 });
notificationSchema.index({ userId: 1, read: 1 });
notificationSchema.index({ type: 1, createdAt: -1 });

export default mongoose.model('Notification', notificationSchema);
