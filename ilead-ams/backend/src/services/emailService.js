import nodemailer from 'nodemailer';
import * as emailTemplates from '../utils/emailTemplates.js';

/**
 * Email Service - Handles all email communications
 */
class EmailService {
  constructor() {
    this.transporter = null;
    this.initialize();
  }

  /**
   * Initialize email transporter
   */
  initialize() {
    const smtpConfig = {
      host: process.env.SMTP_HOST || 'smtp.gmail.com',
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: process.env.SMTP_SECURE === 'true' || false,
      auth: {
        user: process.env.SMTP_USER || '',
        pass: process.env.SMTP_PASS || ''
      }
    };

    if (smtpConfig.auth.user && smtpConfig.auth.pass) {
      this.transporter = nodemailer.createTransport(smtpConfig);
      console.log('📧 Email service initialized');
    } else {
      console.warn('⚠️  Email service disabled - SMTP credentials not configured');
    }
  }

  /**
   * Send email
   */
  async sendEmail(to, subject, htmlContent, textContent = null) {
    if (!this.transporter) {
      console.warn('Email service not configured');
      return false;
    }

    try {
      const mailOptions = {
        from: process.env.SMTP_FROM || process.env.SMTP_USER,
        to,
        subject,
        html: htmlContent,
        text: textContent || htmlContent.replace(/<[^>]*>/g, '')
      };

      const info = await this.transporter.sendMail(mailOptions);
      console.log('✅ Email sent:', info.messageId);
      return true;
    } catch (error) {
      console.error('❌ Error sending email:', error);
      return false;
    }
  }

  /**
   * Send hour log submission confirmation
   */
  async sendHourSubmissionConfirmation(user, hourLog) {
    const html = emailTemplates.hourSubmissionConfirmation(user, hourLog);
    return this.sendEmail(
      user.email,
      'Hour Log Submission Confirmation - I-LEAD AMS',
      html
    );
  }

  /**
   * Send hour log approval notification
   */
  async sendHourApprovalNotification(user, hourLog, supervisor) {
    const html = emailTemplates.hourApprovalNotification(user, hourLog, supervisor);
    return this.sendEmail(
      user.email,
      'Your Hour Log Has Been Approved - I-LEAD AMS',
      html
    );
  }

  /**
   * Send hour log rejection notification
   */
  async sendHourRejectionNotification(user, hourLog, supervisor, reason) {
    const html = emailTemplates.hourRejectionNotification(user, hourLog, supervisor, reason);
    return this.sendEmail(
      user.email,
      'Hour Log Requires Revision - I-LEAD AMS',
      html
    );
  }

  /**
   * Send pending approvals digest to supervisor
   */
  async sendPendingApprovalsDigest(supervisor, pendingLogs) {
    const html = emailTemplates.pendingApprovalsDigest(supervisor, pendingLogs);
    return this.sendEmail(
      supervisor.email,
      `${pendingLogs.length} Pending Hour Log Approvals - I-LEAD AMS`,
      html
    );
  }

  /**
   * Send user account created notification
   */
  async sendAccountCreatedNotification(user, tempPassword) {
    const html = emailTemplates.accountCreatedNotification(user, tempPassword);
    return this.sendEmail(
      user.email,
      'Your I-LEAD AMS Account Has Been Created',
      html
    );
  }

  /**
   * Send password reset email
   */
  async sendPasswordResetEmail(user, resetToken) {
    const resetUrl = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;
    const html = emailTemplates.passwordResetEmail(user, resetUrl);
    return this.sendEmail(
      user.email,
      'Password Reset Request - I-LEAD AMS',
      html
    );
  }

  /**
   * Send weekly summary to apprentice
   */
  async sendWeeklySummary(user, weeklySummary) {
    const html = emailTemplates.weeklySummary(user, weeklySummary);
    return this.sendEmail(
      user.email,
      'Your Weekly Summary - I-LEAD AMS',
      html
    );
  }

  /**
   * Send admin alert
   */
  async sendAdminAlert(admin, alertType, details) {
    const html = emailTemplates.adminAlert(alertType, details);
    return this.sendEmail(
      admin.email,
      `[ALERT] ${alertType} - I-LEAD AMS`,
      html
    );
  }

  /**
   * Send batch email
   */
  async sendBatchEmail(recipients, subject, htmlContent) {
    const results = [];

    for (const recipient of recipients) {
      try {
        const success = await this.sendEmail(recipient, subject, htmlContent);
        results.push({ recipient, success });
      } catch (error) {
        console.error(`Failed to send email to ${recipient}:`, error);
        results.push({ recipient, success: false, error: error.message });
      }
    }

    return results;
  }

  /**
   * Verify SMTP connection
   */
  async verifyConnection() {
    if (!this.transporter) {
      return { status: 'disabled', message: 'Email service not configured' };
    }

    try {
      await this.transporter.verify();
      return { status: 'connected', message: 'SMTP connection verified' };
    } catch (error) {
      return { status: 'error', message: error.message };
    }
  }
}

export default new EmailService();
